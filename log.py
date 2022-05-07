import sys
import variables

def create_log(args):
    dataset = args.dataset
    lr = args.lr
    L2 = args.l2
    add_additional_feature = args.add_additional_feature
    embedding_type = args.embedding_type
    neg_x = args.neg_x
    sl_lambda = args.sl_lambda
    fusion_type = args.fusion_type
    
    train_type = args.training_type
    
    mode = args.mode
    nb_epoch = args.nb_epoch
    alpha = args.alpha  
    
    prev_k_loss = args.prev_k_loss
    model_type = args.model_type
   
    seed = args.seed

    log_var_order = [
        dataset,
        lr,
        L2,
        add_additional_feature,
        embedding_type,
        fusion_type,
        train_type,
        mode,
        neg_x,
        nb_epoch,
        sl_lambda,
        alpha,
        prev_k_loss,
        seed,
        model_type
    ]

    log_filename = variables.dir_logs + "/{}_{}_{}_addF_{}_{}_f_{}_tt_{}_m_{}_neg_x_{}_e_{}_sl_L_{}_a_{}_pkl_{}_seed_{}_{}.txt".format(
        *log_var_order)

    sys.stdout = sys.stderr = open(log_filename, 'w')
    sys.stdout.flush()


