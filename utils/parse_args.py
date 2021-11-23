import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default='SETR_Naive_S', help='SETR_Naive_S, SETR_Naive_L, SETR_Naive_H, SETR_PUP_S, '
                                                  ' SETR_PUP_L, SETR_PUP_H, SETR_MLA_S, SETR_MLA_L, SETR_MLA_H')
    # ckpt
    parser.add_argument("--ckpt_path", default='.\\ckpt',type=str, help='save directory for check point')
    parser.add_argument("--data_path", default='.\\oct_data', type=str, help='path for data')

    # hyper-parameter
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--ckpt", default=50, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--crop_size", default=(480, 480), type=tuple)


    return parser.parse_args()
