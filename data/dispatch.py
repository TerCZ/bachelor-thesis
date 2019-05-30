import numpy as np


class LR:
    def _add_ones(self, X_train):
        ones = np.ones(shape=(X_train.shape[0], 1))
        return np.concatenate((ones, X_train), axis=1)

    def train(self, X_train, y_train):
        X = self._add_ones(X_train)
        X_t = np.transpose(X)
        inv = np.linalg.pinv(np.dot(X_t, X))
        self.W = np.dot(np.dot(inv, X_t), y_train)

    def predict(self, X):
        X = self._add_ones(X)
        return np.dot(X, self.W)


if __name__ == "__main__":
    data_n = 200000000
    mean = 0
    sigma = 2

    X = np.random.lognormal(mean, sigma, size=(data_n,1))
    X.sort(axis=0)  # in-place sort
    X = X / np.max(X) * 1e9  # scale to 1B as the paper says
    y = np.arange(X.shape[0], dtype=np.int64)

    lr = LR()
    lr.train(X, y)
    sec_stg_model_n = 10000

    dispatch = lr.predict(X) * sec_stg_model_n / data_n
    dispatch = dispatch.astype(np.int).reshape((-1,))
    dispatch[dispatch < 0] = 0
    dispatch[dispatch >= sec_stg_model_n] = sec_stg_model_n - 1
    
    # sec_stage_data_n = np.zeros((sec_stg_model_n,))
    # for model_i in dispatch:
    #     sec_stage_data_n += 1
    
    sec_stage_data_n = []

    dispatch = dispatch.reshape((-1,))
    dispatch.sort()
    
    cur_target_i = dispatch[0]
    cur_start_i = 0
    for cur_i, target_i in enumerate(dispatch):
        if target_i != cur_target_i:
            sec_stage_data_n.append(cur_i - cur_start_i)
            cur_target_i = target_i
            cur_start_i = cur_i
    sec_stage_data_n.append(len(dispatch) - cur_start_i)
    
    sec_stage_data_n = np.asarray(sec_stage_data_n)
    sec_stage_data_n.sort()
    
    np.save("dispatch", sec_stage_data_n)
