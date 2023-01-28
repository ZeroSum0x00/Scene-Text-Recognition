import os
import tensorflow as tf
from tqdm import tqdm
from utils.logger import logger
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class AccuracyEvaluate(tf.keras.callbacks.Callback):
    def __init__(self, 
                 val_dataset, 
                 converter,
                 result_path,
                 show_frequency,
                 save_best=True,
                 verbose=True):
        super(AccuracyEvaluate, self).__init__()
        self.val_dataset          = val_dataset
        self.converter            = converter
        self.result_path          = result_path if result_path else "./saved_weights/"
        self.show_frequency       = show_frequency
        self.save_best            = save_best
        self.verbose              = verbose
        self.accuracys            = [0]
        self.epoches              = [0]
        self.current_accuracy     = 0.0
        
    def _sort_criteria(self, data):
        return data['probability']
    
    def on_epoch_end(self, epoch, logs=None):
        temp_epoch = epoch + 1
        n_correct = 0
        n_dataset = self.val_dataset.N
        list_results = []
        if temp_epoch % self.show_frequency == 0:
            print("\nGet Accuracy.")
            for images, labels, lenghts in tqdm(self.val_dataset):
                preds, preds_length, preds_max_prob = self.model.predict(images)
                
                for i in range(labels.shape[0]):
                    result = {}
                    label     = labels[i]
                    label_idx = lenghts[i]
                    pred      = preds[i]
                    pred_idx  = preds_length[i]
                    prob      =  tf.reduce_mean(preds_max_prob[i])
                    label_str = self.converter.decode(label[tf.newaxis, :], label_idx[tf.newaxis])
                    pred_str  = self.converter.decode(pred[tf.newaxis, :], pred_idx[tf.newaxis])
                    if label_str == pred_str:
                        n_correct += 1
                    if self.verbose:
                        result['label'] = label_str
                        result['predict'] = pred_str
                        result['probability'] = prob.numpy()
                        result['TF'] = "True" if label_str == pred_str else "False"
                        list_results.append(result)
            accuracy = n_correct / float(n_dataset) * 100
            
            if self.verbose:
                list_results.sort(reverse=True, key=self._sort_criteria)
                list_results = list_results[:10]
                
                dashed_line = '-' * 76
                head = f'{"       Ground Truth":25s} | {"       Prediction":25s} | {"Confidence"} | T/F'
                separative_line = '{:<26}+{:<27}+{:<12}+{:<8}'.format('-'*26, '-'*27, '-'*12, '-'*8)

                predicted_result_log = f'{dashed_line}\n{head}\n{separative_line}\n'

                for key, result in enumerate(list_results):
                    gt   = result['label']
                    pr   = result['predict']
                    conf = result['probability']
                    t   =  result['TF']
                    predicted_result_log += f'{gt:25s} | {pr:25s} | {conf:<10.4f} | {t}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                
            print(f'\nCurrent accucary: {accuracy}%')
            if self.save_best:
                if accuracy > self.current_accuracy:
                    logger.info(f'Accuracy score increase {self.current_accuracy:.2f}% to {accuracy:.2f}%')
                    logger.info(f'Save best Accuracy weights to {self.result_path}best_accuracy')                    
                    self.model.save_weights(self.result_path + 'best_accuracy')
                    self.current_accuracy = accuracy
            self.accuracys.append(accuracy)
            self.epoches.append(temp_epoch)
            
            with open(os.path.join(self.result_path, "epoch_accuracy.txt"), 'a') as f:
                if epoch == 0:
                    f.write(f"Accuracy score in epoch 0: 0.0")
                f.write(f"Accuracy score in epoch {epoch + 1}: {str(accuracy)}")
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.accuracys, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('A Accuracy Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.result_path, "epoch_accuracy.png"))
            plt.cla()
            plt.close("all")
