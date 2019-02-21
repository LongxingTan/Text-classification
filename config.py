import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_type', type=str, default='static',
                    help='Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')

parser.add_argument('--dataset', type=str, default='reuters_multilabel_dataset',
                    help='Options: reuters_multilabel_dataset, MR_dataset, SST_dataset')

parser.add_argument('--loss_type', type=str, default='margin_loss',
                    help='margin_loss, spread_loss, cross_entropy')

parser.add_argument('--model_type', type=str, default='capsule-B',
                    help='CNN, KIMCNN, capsule-A, capsule-B')

parser.add_argument('--has_test', type=int, default=1, help='If data has test, we use it. Otherwise, we use CV on folds')
parser.add_argument('--has_dev', type=int, default=1, help='If data has dev, we use it, otherwise we split from train')

parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=25, help='Batch size for training')

parser.add_argument('--use_orphan', type=bool, default='True', help='Add orphan capsule or not')
parser.add_argument('--use_leaky', type=bool, default='False', help='Use leaky-softmax or not')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')#CNN 0.0005
parser.add_argument('--margin', type=float, default=0.2, help='the initial value for spread loss')



args = parser.parse_args()
params = vars(args)

print(params)