import sys
sys.path.append("./src/")
import utils as src_utils

if __name__ == "__main__":
    experiment_dir = "./experiments/baseline"
    args = {'--experiment_dir': experiment_dir,
            '--print_every': 1,
            '--num_epochs': 20,
            '--save_every': 1,
            '--data_path': './data/patches/',
            '--batch_size': 96,
            '--model_save_path': experiment_dir + "/model.pt",
            '--n': 24000,
            '--test_split': 0.1,
            '--val_split': 0.1,
            '--train_split': 0.8,
            '--train_img_directory': experiment_dir + '/images/train_images/',
            '--test_img_directory': experiment_dir + '/images/test_images/',
            '--noise_type': 'gaussian'
            }
    # To retrain uncomment these lines

    ###################################
    # train_losses,train_PSNRs,val_losses,val_PSNRs,best_val_PSNR = train(args)
    # np.save(experiment_dir + '/train_PSNRs.npy', np.array(train_PSNRs))
    # np.save(experiment_dir + '/train_losses.npy', np.array(train_losses))
    # np.save(experiment_dir + '/val_PSNRs.npy', np.array(val_PSNRs))
    # np.save(experiment_dir + '/val_losses.npy', np.array(val_losses))

    # loss = train_losses[-1]
    # print("Final training loss: {}, Best Validation PSNR: {}".format(loss, best_val_PSNR))

    ###################################

    test_loss, test_PSNR, test_SSIM = src_utils.Test(args)
    print("Test Results Gaussian: ")
    print("Test loss: {}, Test PSNR: {}, Test SSIM: {}".format(test_loss, test_PSNR, test_SSIM))

    experiment_dir = "./experiments/mixture"
    args = {'--experiment_dir': experiment_dir,
            '--print_every': 1,
            '--num_epochs': 20,
            '--save_every': 1,
            '--data_path': './data/patches/',
            '--batch_size': 96,
            '--model_save_path': experiment_dir + "/model.pt",
            '--n': 24000,
            '--test_split': 0.1,
            '--val_split': 0.1,
            '--train_split': 0.8,
            '--train_img_directory': experiment_dir + '/images/train_images/',
            '--test_img_directory': experiment_dir + '/images/test_images/',
            '--noise_type': 'mixture'
            }
    # To retrain uncomment these lines

    ###################################
    # train_losses,train_PSNRs,val_losses,val_PSNRs,best_val_PSNR = train(args)
    # np.save(experiment_dir + '/train_PSNRs.npy', np.array(train_PSNRs))
    # np.save(experiment_dir + '/train_losses.npy', np.array(train_losses))
    # np.save(experiment_dir + '/val_PSNRs.npy', np.array(val_PSNRs))
    # np.save(experiment_dir + '/val_losses.npy', np.array(val_losses))

    # loss = train_losses[-1]
    # print("Final training loss: {}, Best Validation PSNR: {}".format(loss, best_val_PSNR))

    ###################################

    test_loss, test_PSNR, test_SSIM = src_utils.Test(args)
    print("Test Results Dependent")
    print("Test loss: {}, Test PSNR: {}, Test SSIM: {}".format(test_loss, test_PSNR, test_SSIM))

    experiment_dir = "./experiments/s_p"
    args = {'--experiment_dir': experiment_dir,
            '--print_every': 1,
            '--num_epochs': 20,
            '--save_every': 1,
            '--data_path': './data/patches/',
            '--batch_size': 96,
            '--model_save_path': experiment_dir + "/model.pt",
            '--n': 24000,
            '--test_split': 0.1,
            '--val_split': 0.1,
            '--train_split': 0.8,
            '--train_img_directory': experiment_dir + '/images/train_images/',
            '--test_img_directory': experiment_dir + '/images/test_images/',
            '--noise_type': 's&p'
            }
    # To retrain uncomment these lines

    ###################################
    # train_losses,train_PSNRs,val_losses,val_PSNRs,best_val_PSNR = train(args)
    # np.save(experiment_dir + '/train_PSNRs.npy', np.array(train_PSNRs))
    # np.save(experiment_dir + '/train_losses.npy', np.array(train_losses))
    # np.save(experiment_dir + '/val_PSNRs.npy', np.array(val_PSNRs))
    # np.save(experiment_dir + '/val_losses.npy', np.array(val_losses))

    # loss = train_losses[-1]
    # print("Final training loss: {}, Best Validation PSNR: {}".format(loss, best_val_PSNR))

    ###################################

    test_loss, test_PSNR, test_SSIM = src_Test(args)
    print("Test Results Impulse")
    print("Test loss: {}, Test PSNR: {}, Test SSIM: {}".format(test_loss, test_PSNR, test_SSIM))
