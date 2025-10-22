import torch

def project_noise(noise, epsilon):
    """
    Projects the noise tensor to have a norm less than or equal to epsilon.

    Args:
        noise: The noise tensor.
        epsilon: The maximum norm allowed.

    Returns:
        The projected noise tensor.
    """
    norm = noise.norm(p=2, dim=(1, 2, 3), keepdim=True)
    norm = torch.clamp(norm, min=epsilon)
    return noise / norm * epsilon


def optimize_images(
    net_list,
    images_train,
    labels_train,
    train_criterion,
    device,
    # pipe,
    num_iterations=100,
    learning_rate=0.01,
    lbd=0.1,
    lbd_diff=0.1,
    bounded_noise=False,
    epsilon=1 / 255,
):
    """
    Optimize images by adding noise to minimize loss.

    Args:
        net: The neural network model.
        images_train: Training images.
        labels_train: Training labels.
        train_criterion: Training criterion.
        device: Device to run the optimization on.
        pipe: Diffusion model pipeline.
        num_iterations: Number of optimization iterations.
        learning_rate: Learning rate for the optimizer.
        lbd: Coefficient for score distillation loss.
        lbd_diff: Coefficient for difficulty loss.
        bounded_noise: Whether to bound the noise.
        epsilon: Epsilon value for bounded noise.

    Returns:
        The images with optimized noise.
    """
    # Set the model to evaluation mode

    for net in net_list:
        net.eval()

    noise_list = []

    kk = 0

    for net in net_list:
        kk += 1
        if kk != 1:
            for i in tqdm(range(num_iterations), desc='num_iterations'):

                # Create a noise tensor and set it to require gradients
                if args.noise_init == 'zero':
                    noise = torch.zeros_like(
                        images_train, requires_grad=True, device=device)
                elif args.noise_init == 'one':
                    noise = torch.ones_like(
                        images_train, requires_grad=True, device=device)
                elif args.noise_init == 'rand':
                    noise = torch.rand_like(
                        images_train, requires_grad=True, device=device)

                # Load diffusion model to the device
                # pipe.to(device)

                # Define the optimizer to optimize the noise
                optimizer = optim.Adam([noise], lr=learning_rate)
                optimizer.zero_grad()
                # Add the noise to the images
                noisy_images = images_train + noise
                outputs = net(noisy_images)
                # Compute score distillation loss
                # all_same_logits = torch.ones_like(outputs) / outputs.shape[1]
                # loss_sd = get_score_distillation_loss(
                #     pipe, noisy_images, steps=args.step)
                # Compute difficulty loss
                # loss_diff = F.kl_div(F.log_softmax(outputs, dim=1), all_same_logits)
                # Total loss
                loss = (
                    train_criterion(outputs, labels_train)  # CrossEntropy
                    # + lbd * loss_sd
                    # + lbd_diff * loss_diff
                )
                # Backward pass
                loss.backward()
                # Update the noise using the optimizer
                optimizer.step()

                # Project the noise to the L2 norm ball if required
                if bounded_noise:
                    noise.data = project_noise(noise.data, epsilon)

                # Reduce learning rate after half iterations
                if i == num_iterations // 2:
                    for g in optimizer.param_groups:
                        g["lr"] = learning_rate / 10
                # Print loss every 10% of iterations
                # if i % (num_iterations // 10) == 0:
                #     print(f"Iteration {i}, Loss: {loss.item()}")
            noise_list.append(noise)

    # Return the images with optimized noise
    optimized_images = images_train + sum(noise_list)
    return optimized_images
import numpy as np
pt_file_path = '/data3/yyt/FD/DC/result_true/CIFAR10/IPC10/model_CIFAR10_ConvNet_10ipc_[MSE_0.5_poolNone_layerNone_CE0.1_nfeat1_usemean_normFalse]_DM_[2024-08-22 17:21:06]_poolingFlagFalse_imgUdpTrue.pt'  
checkpoint = torch.load(pt_file_path)
data_save=checkpoint['data']
print("===========")
print(np.mean(checkpoint['accs_all_exps']['ConvNet']))
for idx, (image_syn, label_syn, feature_syn) in enumerate(data_save):
    print(f" image_syn {idx+1} shape : {image_syn.shape}")
    print(f" label_syn {idx+1} shape : {label_syn.shape}")
    print(f" feature_syn {idx+1} shape : {feature_syn.shape}")

print(label_syn.shape)
    
