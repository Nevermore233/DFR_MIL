import time
from sklearn.model_selection import train_test_split
from model import *
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset600', type=bool, default=True, help='choose dataset')
parser.add_argument('--compare', type=bool, default=False, help='choose dataset')
args = parser.parse_args()


if __name__ == '__main__':
    data_path1 = "data_prepare/output/MIL_600_0630.csv"
    data_path2 = "data_prepare/output/MIL_1000_0630.csv"
    pfs_avg_path = "data_prepare/output/pfs_avg_0630.csv"
    ma_avg_path = "data_prepare/output/ma_avg_0630.csv"
    if args.dataset600:
        data = pd.read_csv(data_path1, index_col=0, encoding="utf-8")
    else:
        data = pd.read_csv(data_path2, index_col=0, encoding="utf-8")

    pfs_avg = pd.read_csv(pfs_avg_path, index_col=0, encoding="utf-8")
    ma_avg = pd.read_csv(ma_avg_path, index_col=0, encoding="utf-8")

    data_X = data.drop(columns=['label', 'bag_labels'])
    df = data.loc[:, ["bag_names", "bag_labels"]].drop_duplicates(['bag_names'])
    bag_names, bag_labels = df.iloc[:, 0], df.iloc[:, 1]
    x_train_all, x_test, y_train_all, y_test = train_test_split(
        bag_names, bag_labels)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_all, y_train_all)

    x_train, x_test, y_train, y_test, x_valid, y_valid = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), np.array(x_valid), np.array(y_valid)

    lr = 0.001
    hidden_num = 50
    epochs = 200
    feature_num = 3

    if not args.compare:
        # HL_loss
        time_start = time.time()
        model, Q = DFR_MIL(lr, hidden_num, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X, ma_avg)
        time_end = time.time()
        time_sum = time_end - time_start
        print("time:", time_sum)
        print('threshold:', Q)

        # Start prediction
        print("-----Start prediction------")
        y_hat = []
        for bag in range(np.shape(x_test)[0]):
            bag_name = x_test[bag]
            test_X = data_X[data_X['bag_names'] == bag_name].drop(columns=['bag_names'])
            pred = predict(model, test_X)
            y_hat.append(pred)

        # Prediction results
        y_hat = pd.DataFrame(y_hat)
        y = pd.DataFrame(y_test)
        res = pd.concat([y_hat, y], axis=1)
        res.columns = ["y_hat", "y"]
        test_mse = mean_squared_error(res.iloc[:, 0], res.iloc[:, 1])
        HL_res = Hosmer_Lemeshow_test(res)
        HL_value = HL_res[0]
        p_val = HL_res[1]
        print("test_set\ntest_mse:{}:HL_value:{},p_val:{}".format(test_mse, HL_value, p_val))
        print("-----End of prediction------")

        if args.dataset600:
            # main_result
            x_test_ = pd.DataFrame(x_test, columns=['bag_names'])
            test_pfs_avg = pd.merge(x_test_, pfs_avg, on='bag_names', how='left').set_index('bag_names')
            test_ma_avg = pd.merge(x_test_, ma_avg, on='bag_names', how='left').set_index('bag_names')
            main_result = pd.concat([test_pfs_avg.reset_index(drop=True), y.reset_index(drop=True), y.reset_index(drop=True), y_hat.reset_index(drop=True), test_ma_avg.reset_index(drop=True),  test_ma_avg.reset_index(drop=True)], axis=1)
            main_result.columns = ['pfs', 'status', 'y', 'y_pred', 'mutation abundance', 'label_2']
            main_result['status'] = main_result['status'].apply(lambda x: 1 if x >= 0.5 else 0)
            main_result['label_2'] = main_result['label_2'].apply(lambda x: 1 if x >= Q else 0)
            main_result['dfr'] = 1 - main_result['y']
            main_result['dfr_pred'] = 1 - main_result['y_pred']

            main_result.to_csv('data_prepare/output/main_result.csv')
            # Then, 'main_result.csv' will be used for survival analysis using GraphPad Prism 8 software.

    #################################### compare ####################################
    if args.compare:
        ###### huber loss
        print('##### huber loss #####')
        model_huber = Model_Building_huber_loss(lr, hidden_num, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X)
        prediction(data_X, x_test, y_test, model_huber)
        ##### mse loss
        print('##### mse loss #####')
        model_mse = Model_Building_mse_loss(lr, hidden_num, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X)
        prediction(data_X, x_test, y_test, model_mse)
        ###### mae loss
        print('##### mae loss #####')
        model_mae = Model_Building_mae_loss(lr, hidden_num, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X)
        prediction(data_X, x_test, y_test, model_mae)
        ##### logcosh loss
        print('##### logcosh loss #####')
        model_logcosh = Model_Building_logcosh_loss(lr, hidden_num, epochs, feature_num, x_train, y_train, x_valid, y_valid, data_X)
        prediction(data_X, x_test, y_test, model_logcosh)