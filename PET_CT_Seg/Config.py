import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training Config")

    #数据集路径(PCLT20K)
    parser.add_argument('--test_PET_path', type=str, default='./PET/')
    parser.add_argument('--test_CT_path', type=str, default='./CT/')
    parser.add_argument('--test_mask_path', type=str, default='./mask/')


    return parser.parse_args()
