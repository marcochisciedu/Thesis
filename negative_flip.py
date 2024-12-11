import torch

# Calculates regular negative flip rate 
def negative_flip_rate(model_v1, model_v2, test_loader, dict_output = False):
    negative_flips= 0
    total = 0
    flips ={}

    model_v1.eval()
    model_v2.eval()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data

            # Get both models outputs
            if dict_output:
                logits_v1 = model_v1(inputs)['logits']
                logits_v2 = model_v2(inputs)['logits']
            else:
                logits_v1 = model_v1(inputs)
                logits_v2 = model_v2(inputs)

            output_v1 = logits_v1.argmax(1)
            output_v2 = logits_v2.argmax(1)

            # Negative flip if model_v2 is wrong while model_v1 is correct
            flipping = torch.logical_and(output_v2 != labels, output_v1 == labels)
            negative_flips += (flipping).sum().item()
            total += inputs.size(0)

            # For each negative flip store and count each flip
            flip_indices = flipping.nonzero()
            for index in flip_indices:
                if (output_v1[index].item(), output_v2[index].item()) in flips:
                    flips[(output_v1[index].item(), output_v2[index].item())] += 1
                else: 
                    flips[(output_v1[index].item(), output_v2[index].item())] = 1

    return (negative_flips/total), flips, negative_flips

# Calculates an improved version of NFR, accounts for flips when both models are incorrect
def improved_negative_flip_rate(model_v1, model_v2, test_loader, dict_output = False):
    negative_flips= 0
    total = 0
    flips = {}

    model_v1.eval()
    model_v2.eval()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data

            # Get both models outputs
            if dict_output:
                logits_v1 = model_v1(inputs)['logits']
                logits_v2 = model_v2(inputs)['logits']
            else:
                logits_v1 = model_v1(inputs)
                logits_v2 = model_v2(inputs)

            output_v1 = logits_v1.argmax(1)
            output_v2 = logits_v2.argmax(1)
            
            # Negative flip if model_v2 is incorrect and its prediction is not the same as model_v1
            flipping = torch.logical_and(output_v2 != labels,output_v2 != output_v1) 
            negative_flips += (flipping).sum().item()
            total += inputs.size(0)

             # For each negative flip store and count each flip
            flip_indices = flipping.nonzero()
            for index in flip_indices:
                if (output_v1[index].item(), output_v2[index].item()) in flips:
                    flips[(output_v1[index].item(), output_v2[index].item())] += 1
                else: 
                    flips[(output_v1[index].item(), output_v2[index].item())] = 1
    
    return (negative_flips/total), flips, negative_flips

# Calculate the relative NFR given a NFR (normal or improved) and the accuracies 
# of the two compared models
def relative_negative_flip_rate(neg_flip_rate, accuracy_old, accuracy_new):
    return neg_flip_rate/(accuracy_old*(1-accuracy_new))