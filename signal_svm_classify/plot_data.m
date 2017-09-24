clear all;

% ��������
label = importdata('startbit.txt');
feature = importdata('accept.txt');
label = reshape(label,1,10000);
feature = reshape(feature,1,10000);

% ��һά�ֲ�
x = 1:10000;
y = feature(1:10000);
figure(1);
scatter(x,y,10,label,'filled');

% ����ά����
x = feature(1:9999);
y = feature(2:10000);
c = label(2:10000);
figure(2);
scatter(x,y,10,c,'filled');

% ����ά����
x = feature(1:9998);
y = feature(2:9999);
z = feature(3:10000);
c = label(3:10000);
figure(3);
scatter3(x,y,z,10,c,'filled');

% ��ά��ǰ�����ݼ��Լ���Ч�����
x = feature(1:9998);
y = feature(2:9999);
z = feature(3:10000);
c = label(2:9999);
figure(4);
scatter3(x,y,z,10,c,'filled');