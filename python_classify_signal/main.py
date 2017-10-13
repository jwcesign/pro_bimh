# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
import os

def find_start(zero_index,one_index):
    if zero_index > one_index:
        if zero_index > 32:
            while zero_index > 32:
                zero_index -= 32
            return zero_index
        else:
            return zero_index
    else:
        if one_index > 32:
            while one_index > 32:
                one_index -= 32
            return one_index
        else:
            return one_index

def sync_data(start_index,feature,label):
    feature = feature.tolist()
    label = label.tolist()
    feature_length = len(feature)
    label_length = feature_length/32
    feature_solve = []
    label_solve = []
    for i in range(0,label_length):
        if (i+1)*32+start_index > feature_length:
            return np.array(feature_solve),np.array(label_solve)
        else:
            feature_solve.append(np.array(feature[i*32+start_index:(i+1)*32+start_index]))
            label_solve.append(np.array(label[i*32+start_index:(i+1)*32+start_index]))

def get_ber(feature,label,required_dimension,number_train,c):
    dimension_parameter = required_dimension/2
    feature_tmp = feature.reshape(1,len(feature)*32)
    # 训练集
    x_train_tm = feature_tmp[0][32-dimension_parameter:(number_train+1)*32+dimension_parameter]
    x_test_tm = feature_tmp[0][10000*32-dimension_parameter:120000*32+dimension_parameter]
    y_train = label[0][1:number_train+1]
    y_test = label[0][10000:120000]
    num_err_re = []
    for i in range(0,32):
        x_train = []
        x_test = []
        for j in range(0,number_train):
            x_train.append(x_train_tm[j*32+i:j*32+required_dimension+i])
        x_train = np.array(x_train)
        for j in range(0,110000):
            x_test.append(x_test_tm[j*32+i:j*32+required_dimension+i])
        x_test = np.array(x_test)
        clf = svm.SVC(C=c,kernel='linear')
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        clf.fit(x_train,y_train)
        py = clf.predict(x_test)
        num_err = sum(py != y_test)
        num_err_re.append(num_err/110000.0)

    return np.array(num_err_re).min()

def svm_model(feature,label,number_train,choice,c):
    # 32d
    if choice == 3:
        # 训练集
        x_train = feature[0:number_train,:]
        y_train = label[0][0:number_train]
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        y_train = np.array(y_train)
        # 测试集
        x_test = feature[10000:120000,:]
        y_test = label[0][10000:120000]
        x_test = scaler.transform(x_test)
        clf = svm.SVC(kernel='linear',C=c)
        clf.fit(x_train,y_train)
        py = clf.predict(x_test)
        error_number = sum(py != y_test)
        return error_number/110000.0

    # 92d
    if choice == 4:
        # 训练集
        x_train = []
        y_train = label[0][1:number_train+1]
        for i in range(0,number_train):
            x_train.append(feature[i,:].tolist()+feature[i+1,:].tolist()+feature[i+2,:].tolist())
        x_train = np.array(x_train)
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        # 测试集
        x_test = []
        y_test = label[0][10001:120001]
        for i in range(10000,120000):
            x_test.append(feature[i,:].tolist()+feature[i+1,:].tolist()+feature[i+2,:].tolist())
        x_test = np.array(x_test)
        x_test = scaler.transform(x_test)
        clf = svm.SVC(kernel='linear',C=c)
        clf.fit(x_train,y_train)
        py = clf.predict(x_test)
        error_number = sum(py != y_test)
        return error_number/110000.0

    # 5d
    if choice == 7:
        return get_ber(feature,label,5,number_train,c)
    # 7d
    if choice == 9:
        return get_ber(feature,label,7,number_train,c)
    # 9d
    if choice == 10:
        return get_ber(feature,label,9,number_train,c)

def main():
    # 导入平均计算的ber，前10000个以备作为训练数据
    ber_total = [[],[],[],[],[],[],[],[],[],[],[],[]]
    number_train = 600
    c_parameter = 0.9
    for i in range(1,11):
        print '-'*10+str(i)+'-'*10
        ber_mean = np.loadtxt('./data_1/20_'+str(i)+'ber.txt',delimiter=',')
        ber_eq_mean = np.loadtxt('./data_2/20_'+str(i)+'ber.txt',delimiter=',')
        # 导入波形数据
        feature = np.loadtxt('./data_1/20_'+str(i)+'r.txt',delimiter=',')
        feature_eq = np.loadtxt('./data_2/20_'+str(i)+'r.txt',delimiter=',')
        # 导入电平数据
        label = np.loadtxt('./data_1/20_'+str(i)+'s.txt',delimiter=',')
        rows,columns = label.shape
        label_tmp = label.reshape(1,rows*columns).tolist()
        label_tmp = [int(j) for j in label_tmp[0]]
        label = np.array(label_tmp).reshape(32,columns)

        label_eq = np.loadtxt('./data_2/20_'+str(i)+'s.txt',delimiter=',')
        rows_eq,columns_eq = label_eq.shape
        label_eq_tmp = label_eq.reshape(1,rows_eq*columns_eq).tolist()
        label_eq_tmp = [int(j) for j in label_eq_tmp[0]]
        label_eq = np.array(label_eq_tmp).reshape(32,columns_eq)



        # 处理数据
        feature = feature.reshape(1,rows*columns)[0]
        label = label.transpose().reshape(1,rows*columns)[0]

        feature_eq = feature_eq.reshape(1,rows_eq*columns_eq)[0]
        label_eq = label_eq.transpose().reshape(1,rows_eq*columns_eq)[0]

        # 提取需要的数据（找到开始的头）
        zero_index = label.tolist().index(0)
        one_index = label.tolist().index(1)
        start_index = find_start(zero_index,one_index)

        zero_index_eq = label_eq.tolist().index(0)
        one_index_eq = label_eq.tolist().index(1)
        start_index_eq = find_start(zero_index_eq,one_index_eq)

        # 获取需要格式化后的数据
        feature,label = sync_data(start_index,feature,label)
        feature_eq,label_eq = sync_data(start_index_eq,feature_eq,label_eq)

        # 构造需要的数据格式
        label_tmp = label[:,1].reshape(1,columns-1)
        label_tmp_eq = label_eq[:,1].reshape(1,columns_eq-1)

        # 训练模型,并测试BER
        print '代码中的旧方法：'+str(ber_mean.min())
        ber_total[0].append(ber_mean.min())

        print '代码中的新方法(eq)：'+str(ber_eq_mean.min())
        ber_total[1].append(ber_eq_mean.min())

        tmp = svm_model(feature,label_tmp,number_train,3,c_parameter)
        print '数据同步过后用svm方法(32d)：'+str(tmp)
        ber_total[2].append(tmp)

        tmp = svm_model(feature_eq,label_tmp_eq,number_train,3,c_parameter)
        print '数据同步过后用svm方法(32d,eq)：'+str(tmp)
        ber_total[3].append(tmp)

        tmp = svm_model(feature,label_tmp,number_train,4,c_parameter)
        print '数据同步过后用svm方法(96d)：'+str(tmp)
        ber_total[4].append(tmp)

        tmp = svm_model(feature_eq,label_tmp_eq,number_train,4,c_parameter)
        print '数据同步过后用svm方法(96d,eq)：'+str(tmp)
        ber_total[5].append(tmp)

        tmp = svm_model(feature,label_tmp,number_train,7,c_parameter)
        print '解决跨界后的每个点用svm(5d):'+str(tmp)
        ber_total[6].append(tmp)

        tmp = svm_model(feature_eq,label_tmp_eq,number_train,7,c_parameter)
        print '解决跨界后的每个点用svm(5d,eq):'+str(tmp)
        ber_total[7].append(tmp)

        tmp = svm_model(feature,label_tmp,number_train,9,c_parameter)
        print '解决跨界后的svm(7d):'+str(tmp)
        ber_total[8].append(tmp)

        tmp = svm_model(feature_eq,label_tmp_eq,number_train,9,c_parameter)
        print '解决跨界后的svm(7d,eq):'+str(tmp)
        ber_total[9].append(tmp)

        tmp = svm_model(feature,label_tmp,number_train,10,c_parameter)
        print '解决跨界后的svm(9d):'+str(tmp)
        ber_total[10].append(tmp)

        tmp = svm_model(feature_eq,label_tmp_eq,number_train,10,c_parameter)
        print '解决跨界后的svm(9d,eq):'+str(tmp)
        ber_total[11].append(tmp)

    np.savetxt('final.txt',ber_total)


def plot_data():
    ber = np.loadtxt('final.txt',delimiter=' ')
    k = range(1,11)
    for i in range(0,12):
        if i == 0:
            plt.plot(k,np.log10(ber[i,:]),label='123')
        else:
            plt.plot(k,np.log10(ber[i,:]))


    plt.show()

# 运行程序
if __name__ == '__main__':
    if os.path.exists('final.txt'):
        plot_data()
    else:
        main()
        plot_data()
