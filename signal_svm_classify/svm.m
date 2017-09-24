clear all;
label = importdata('startbit.txt');
feature = importdata('accept3.txt');

% ѵ����
x_train = zeros(6000,3);
y_train = zeros(1,6000);
for i=3:6002
    x_train(i-2,:) = feature(i-2:i);
end
y_train = label(2:6001);

% ���Լ�
x_test = zeros(3000,3);
y_test = zeros(1,3000);
for i=6003:9002
    x_test(i-6002,:)=feature(i-2:i);
end
y_test = label(6002:9001);

% ѵ��SVM������
svmStr = svmtrain(x_train,y_train,'kernel_function','rbf');

% ���Է�����
py = svmclassify(svmStr,x_test);

% ������ȷ��
c_r = 0;
for i=1:3000
    if py(i) == y_test(i)
        c_r = c_r+1;
    end
end

% ��ȷ��
c_r/3000

% 3:0.5587; 2:0.5457; 4:0.5803; 5:0.5710(��ǰ��4άЧ�����)
% ǰ���Լ�,Ч�����