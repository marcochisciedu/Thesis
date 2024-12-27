"""
Fast and self-contained training script for CIFAR-10.
"""

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import wandb
import argparse, yaml
from math import ceil
from dotenv import load_dotenv

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip import *
from NFR_losses import *
from fixed_classifiers import *
torch.backends.cudnn.benchmark = True

# We express the main training hyperparameters (batch size, learning rate, momentum, and weight decay)
# in decoupled form, so that each one can be tuned independently. This accomplishes the following:
# * Assuming time-constant gradients, the average step size is decoupled from everything but the lr.
# * The size of the weight decay update is decoupled from everything but the wd.
# In constrast, normally when we increase the (Nesterov) momentum, this also scales up the step size
# proportionally to 1 + 1 / (1 - momentum), meaning we cannot change momentum without having to re-tune
# the learning rate. Similarly, normally when we increase the learning rate this also increases the size
# of the weight decay, requiring a proportional decrease in the wd to maintain the same decay strength.
#
# The practical impact is that hyperparameter tuning is faster, since this parametrization allows each
# one to be tuned independently. See https://myrtle.ai/learn/how-to-train-your-resnet-5-hyperparameters/.

hyp = {
    'opt': {
        'train_epochs': 9.9,
        'batch_size': 1024,
        'lr': 11.5,                 # learning rate per 1024 examples
        'momentum': 0.85,
        'weight_decay': 0.0153,     # weight decay per 1024 examples (decoupled from learning rate)
        'bias_scaler': 64.0,        # scales up learning rate (but not weight decay) for BatchNorm biases
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3,    # how many epochs to train the whitening layer bias before freezing
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'widths': {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1/9,
        'tta_level': 2,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
        'feat_dim' : 3,         # features' dimension
    },
    'data': {
        'percentage':100,           # percentage of CIFAR10 to use
        'low_classes': None,        # which classes get a lower percentage of data
        'low_percentage': 10,       # percentage of the class to take
    },
    'nfr' : False,                  # if, at the end of each epoch, the NFR will be calculated
    'old_model_name' : None,        # the name of the worse older model that is going to be used in NFR calculation
    'loss' : 'default' ,            # training loss used
    'dSimplex': False,              # if the classifier is a dSimplex
    'fd' :{                         # focal distillation parameters
        'fd_alpha' : 1,
        'fd_beta' : 5,
        'focus_type' : 'old_correct',
        'distillation_type' : 'kl',
        'kl_temperature' : 100,
        'lambda' : 1,
    },
}

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

# First random inputs flip
def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None, drop_last=None, shuffle=None, gpu=0, 
                 percentage = 100, list_low_classes = None, low_percentage = 10):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device(gpu))
        
        # Check if the whole dataset is used or not
        if (percentage >= 100 or percentage<0) and list_low_classes is None:
            self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        else:
            num_imgs = int(data['images'].size()[0]*percentage/100)
            imgs_per_label = [int(num_imgs/len(data['classes']))]*(len(data['classes']))
            # check if some classes need a different percentage of images
            if list_low_classes is not None:
                for index, value in enumerate(list_low_classes):
                    imgs_per_label[value] = int(data['images'].size()[0]*low_percentage[index]/(100*len(data['classes'])))
                    print(f"Using {low_percentage[index]}% of the {value}° class's images")
            print(f"Using 100% of the other classes' images")
            self.images= torch.empty((0,data['images'].size()[1], data['images'].size()[2], data['images'].size()[3]),dtype=torch.uint8, device=gpu)
            self.labels=torch.empty((0),dtype=torch.uint8, device=gpu)
            for i in range(len(data['images'])):
                if imgs_per_label[data['labels'][i]] > 0:
                    self.labels = torch.cat((self.labels,data['labels'][i].reshape(1)), 0)
                    self.images= torch.cat((self.images,data['images'][i].unsqueeze(0) ),0)
                    imgs_per_label[data['labels'][i]] -= 1
                elif all(num == 0 for num in imgs_per_label): 
                    break
        
            self.classes = data['classes']

        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {} # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images['norm'] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1
        
        # Create random batches
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################

def make_net(feat_dim =3):
    widths = hyp['net']['widths']
    batchnorm_momentum = hyp['net']['batchnorm_momentum']
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1'], batchnorm_momentum),
        ConvGroup(widths['block1'], widths['block2'], batchnorm_momentum),
        ConvGroup(widths['block2'], widths['block3'], batchnorm_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], feat_dim, bias=False),
        nn.Linear(feat_dim, 10, bias=False),  # added linear layer to bottleneck the softmax layer (unargmaxability)
        Mul(hyp['net']['scaling_factor']),
    )
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

#############################################
#       Whitening Conv Initialization       #
#############################################

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

############################################
#                Lookahead                 #
############################################

class LookaheadState:
    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}

    def update(self, net, decay):
        for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):

    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, 'reflect')
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#            Easier model loading          #
############################################

def load_trained_model(model_path, wandb_run, feat_dim):
    """
    Load a trained model from a given path.
    """
    # Replace with your model architecture
    model = make_net(feat_dim)
    artifact = wandb_run.use_artifact(WANDB_PROJECT+model_path, type='model')
    artifact_dir = artifact.download()
    model.load_state_dict(torch.load(artifact_dir+'/model.pth'))
    model.eval()
    return model
       
############################################
#                Training                  #
############################################

def main(model_name, run):
    print("Running "+ str(run+1) +"° training")
    # Initialize wandb
    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = model_name+"_CIFAR10_"+ str(hyp['data']['percentage'])+"/" + str(hyp['data']['low_percentage'])+ "percent_"
         +str(hyp['opt']['train_epochs'])+ "epochs",
        config=hyp)
    
    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    # Assuming gradients are constant in time, for Nesterov momentum, the below ratio is how much
    # larger the default steps will be than the underlying per-example gradients. We divide the
    # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
    # of the choice of momentum.
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale # un-decoupled learning rate for PyTorch SGD
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'], percentage = hyp['data']['percentage'],
                                list_low_classes= hyp['data']['low_classes'], low_percentage= hyp['data']['low_percentage'])
    if run == 'warmup':
        # The only purpose of the first run is to warmup, so we can use dummy data
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_train_steps = ceil(len(train_loader) * epochs)

    model = make_net(hyp['net']['feat_dim'])
    if hyp['dSimplex']:
        fixed_weights = dsimplex(num_classes=10)
        model[8].weight.requires_grad = False  # set no gradient for the fixed classifier
        model[8].weight.copy_(fixed_weights)   # set the weights for the classifier

    current_steps = 0
    if hyp['nfr']:
        old_model = load_trained_model(hyp['old_model_name'], wandb_run, hyp['net']['feat_dim'])

    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.23)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (step - warmup_steps) / warmdown_steps
            return 1.0 * (1 - frac) + 0.07 * frac
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
    lookahead_state = LookaheadState(model)

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    # Initialize the whitening layer using training images
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    for epoch in range(ceil(epochs)):

        model[0].bias.requires_grad = (epoch < hyp['opt']['whiten_bias_epochs'])

        ####################
        #     Training     #
        ####################

        starter.record()

        model.train()
        for inputs, labels in train_loader:

            outputs = model(inputs)
            if hyp['loss'] == 'Focal Distillation':
                # Get old model's prediction
                old_outputs = old_model(inputs)
                fd_loss= FocalDistillationLoss(hyp['fd']['fd_alpha'], hyp['fd']['fd_beta'], hyp['fd']['focus_type'],
                                               hyp['fd']['distillation_type'], hyp['fd']['kl_temperature'] )
                loss_focal_distillation = fd_loss(outputs, old_outputs, labels)
                loss_CE = loss_fn(outputs, labels).sum()
                loss = loss_CE + hyp['fd']['lambda']*loss_focal_distillation
            else:
                loss = loss_fn(outputs, labels).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_steps += 1

            if current_steps % 5 == 0:
                lookahead_state.update(model, decay=alpha_schedule[current_steps].item())

            if current_steps >= total_train_steps:
                if lookahead_state is not None:
                    lookahead_state.update(model, decay=1.0)
                break

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        train_loss = loss.item() / batch_size
        val_acc = evaluate(model, test_loader, tta_level=0)
        train_metrics = {'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc}
        wandb.log({**train_metrics}, step= epoch)


        ###################
        #  NFR Evaluation #
        ###################
        if hyp['nfr']:
            nfr, _, _ = negative_flip_rate(old_model, model, test_loader)
            impr_nfr, _ , _ = improved_negative_flip_rate(old_model, model, test_loader)
            print(f"Negative flip rate at epoch {epoch}: {nfr}")
            print(f"Improved negative flip rate at epoch {epoch}: {impr_nfr}")
            wandb.log({'NFR':nfr, 'Improved NFR': impr_nfr}, step= epoch)

    ####################
    #  TTA Evaluation  #
    ####################

    starter.record()
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)
    
    eval_metrics = {'tta_val_acc': tta_val_acc, 'total_time_seconds': total_time_seconds}
    wandb.log({**eval_metrics})
    
    # Save the model on weights and biases as an artifact
    model_artifact = wandb.Artifact(
                 model_name+"_CIFAR10_"+ str(hyp['data']['percentage'])+"_" + str(hyp['data']['low_percentage'])[1:-1].replace(" ", "").replace(",", "_")
                 + "percent_"+str(hyp['opt']['train_epochs'])+ "epochs", type="model",
                description="model trained on run "+ str(run),
                metadata=dict(hyp))

    torch.save(model.state_dict(), "model.pth")
    model_artifact.add_file("model.pth")
    wandb.save("model.pth")
    wandb_run.log_artifact(model_artifact)

    # Close wandb run
    wandb.finish()

    return tta_val_acc

if __name__ == "__main__":
    # Get env variables
    load_dotenv()

    # Setup wandb
    WANDB_USERNAME = os.getenv('WANDB_USERNAME')
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    wandb.login(key= WANDB_API_KEY, verify=True)

    # Select config
    parser = argparse.ArgumentParser(description='Training CIFAR10')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        default=os.path.join(os.getcwd(), "configs/half_CIFAR10.yaml"), 
                        type=str)
    params = parser.parse_args()

    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    model_name = loaded_params['model_name']
    num_models = loaded_params['num_models']
    # Modify the default hyperparameters
    hyp['data']['percentage'] = loaded_params['percentage']
    hyp['opt']['train_epochs'] = loaded_params['epochs']
    hyp['data']['low_classes'] = loaded_params['low_class_list']
    hyp['data']['low_percentage'] = loaded_params['low_percentage']
    hyp['nfr'] = loaded_params['nfr']
    hyp['net']['feat_dim'] = loaded_params['feat_dim']
    hyp['dSimplex'] = loaded_params['dSimplex']
    if hyp['nfr'] == True:
        hyp['old_model_name'] = loaded_params['old_model']
    hyp['loss'] = loaded_params['loss']
    if hyp['loss'] == 'Focal Distillation':
        hyp['fd']['fd_alpha'] = loaded_params['fd_alpha']
        hyp['fd']['fd_beta'] = loaded_params['fd_beta']
        hyp['fd']['focus_type'] = loaded_params['focus_type']
        hyp['fd']['distillation_type'] = loaded_params['distillation_type']
        hyp['fd']['kl_temperature'] = loaded_params['kl_temperature']
        hyp['fd']['lambda'] = loaded_params['lambda']

    # How many times the main is runned
    accs = torch.tensor([main(model_name, run) for run in range(num_models)])

    # Log mean and std
    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "Final log "+  str(hyp['data']['percentage'])+"/" + str(hyp['data']['low_percentage'])+ "percent_"
                +str(hyp['opt']['train_epochs'])+ "epochs",
        config=hyp)
    final_metrics = {'mean': accs.mean(), 'std': accs.std()}
    wandb.log({**final_metrics})
