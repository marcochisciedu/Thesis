import torch

# Calculates regular negative flip rate 
def negative_flip_rate(model_v1, model_v2, test_loader):
    negative_flips= 0
    total = 0

    model_v1.eval()
    model_v2.eval()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data

            # Get both models outputs
            logits_v1 = model_v1(inputs)
            output_v1 = logits_v1.argmax(1)

            logits_v2 = model_v2(inputs)
            output_v2 = logits_v2.argmax(1)

            # Negative flip if model_v2 is wrong while model_v1 is correct
            negative_flips += (torch.logical_and(output_v2 != labels, output_v1 == labels)).sum().item()
            total += inputs.size(0)
    
    return (negative_flips/total)

# Calculates an improved version of NFR, accounts for flips when both models are incorrect
def improved_negative_flip_rate(model_v1, model_v2, test_loader):
    negative_flips= 0
    total = 0

    model_v1.eval()
    model_v2.eval()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data

            # Get both models outputs
            logits_v1 = model_v1(inputs)
            output_v1 = logits_v1.argmax(1)

            logits_v2 = model_v2(inputs)
            output_v2 = logits_v2.argmax(1)
            
            # Negative flip if model_v2 is incorrect and its prediction is not the same as model_v1
            negative_flips += (torch.logical_and(output_v2 != labels,output_v2 != output_v1 )).sum().item()
            total += inputs.size(0)
    
    return (negative_flips/total)