import os
import argparse
import pandas as pd
import torch
from conf import settings
from utils.model_utils import get_network
from utils.data_utils import get_test_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_best_checkpoint():
    root = settings.CHECKPOINT_PATH + '/' + args.net
    best_pth_list = []
    for pth in os.listdir(root):
        if pth.split('-')[-1].split('.')[0] == 'best':
            best_pth_list.append(pth)
    best_pth_list.sort()
    return os.path.join(root, best_pth_list[-1])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-root', type=str, default='/cache/', help='root path')
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-weights', type=str, default=None, help='the weights file you want to test')
    parser.add_argument('-bs', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args, pretrained=False)

    test_csv = pd.read_csv('/cache/test.csv')

    Ocean_test_loader = get_test_dataloader(
        img_path=args.root+'data/',
        test_csv=test_csv,
        num_workers=4,
        batch_size=args.bs,
    )

    if args.weights is None:
        args.weights = get_best_checkpoint()
    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    ans_file = []
    ans_pred = []
    with torch.no_grad():
        for _, [img, fileid] in enumerate(Ocean_test_loader):
            inputs = img.to(device)
            outputs = net(inputs)

            ans_file.extend(fileid)
            ans_pred.extend(outputs.max(1)[1].detach().cpu().numpy())
    ans = [[ans_file[i], ans_pred[i]] for i in range(len(ans_file))]
    ans = pd.DataFrame(ans, columns=['FileID', 'SpeciesID'])
    ans.to_csv('ans.csv', index=None)
    # mox.file.copy('ans.csv', os.path.join(Context.get_output_path(), 'ans.csv'))
    print('save over')