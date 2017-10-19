import keras.callbacks as KCallBacks
from sklearn.metrics import roc_auc_score
import keras.backend as K

class SGDLearningRateTracker(KCallBacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        # print('\nLR: {:.6f}\n'.format(lr))

class roc_callback(KCallBacks.Callback):
    def __init__(self, train_in, train_out, val_in, val_out):
        self.x = train_in
        self.y = train_out
        self.x_val = val_in
        self.y_val = val_out

    def on_train_begin(self, logs=None):
        self.best_auc = -1.0
        self.best_weights = self.model.get_weights()
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict_proba(self.x, verbose=0)
        auc = roc_auc_score(self.y[:, 0], y_pred[:, 0])

        y_pred_val = self.model.predict_proba(self.x_val, verbose=0)
        auc_val = roc_auc_score(self.y_val[:, 0], y_pred_val[:, 0])

        total_auc = (auc_val + auc) / 2
        if total_auc > self.best_auc:
            self.best_auc = total_auc
            self.best_weights = self.model.get_weights()
            # print("Best updated to: %f" % self.best_auc)

        # print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(auc, 4)), str(round(auc_val, 4))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return

class BestRecorder(KCallBacks.Callback):
    def on_train_begin(self, logs={}):
        self.best_loss = 1000000
        self.best_weights = self.model.get_weights()

    # def on_batch_end(self, batch, logs=None):
    #     self.loss.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')
        loss = logs.get('loss')
        total_loss = (val_loss + loss)/2
        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.best_weights = self.model.get_weights()
            print("Best updated to: %f" %self.best_loss)

