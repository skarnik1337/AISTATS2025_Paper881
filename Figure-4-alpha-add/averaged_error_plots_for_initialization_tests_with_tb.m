
clear all
clc
colorStr= ["b","g","c","m","r","y","k"];


alpha=[10^(-2),10^(-4),10^(-6), 10^(-8)]; 

alphaRange= [2 4 6 8];       % overparam
testNumberRange=[1 2 3 4 5];
numIter= 4000;
r=200;
n= 10;
k= 4;
rStar=3;


testNumRangeName=["test_1", "test_2","test_3", "test_4", "test_5"]; 
alphaRangeName= ["minus_2", "minus_4","minus_6","minus_8"]; 


nu=1e-5;

logArum=1/nu.*log(1./(alpha*2000));
numIterSort=floor(3999/(max(logArum))*logArum)+1;


numIterSort(1)=1;

rNumRange=[1 2 3 4 5];

makerType= ["-o","-v","-s","-d"];


%% test error


 trainErrorAverArray=zeros(1,length(alphaRange));
 trainErrorAverArrayMax=zeros(1,length(alphaRange));
 trainErrorAverArrayMin=zeros(1,length(alphaRange));

 testErrorAverArray=zeros(1,length(alphaRange));
 testErrorAverArrayMax=zeros(1,length(alphaRange));
 testErrorAverArrayMin=zeros(1,length(alphaRange));

for alphaNum=1:length(alphaRange)

    alphaName=alphaRangeName(alphaNum);

    clear testErrorArray trainErrorArray;

         testErrorAver=0;
         trainErrorAver = 0;
         testErrorArray=[];
         trainErrorArray=[];

         for test=testNumberRange

            testName=testNumRangeName(test);
  
% test error
            testErrorName=strcat('test_error_',alphaName,'_', testName,'.mat');
            testError=load(testErrorName);
           
           % testErrorAver=testErrorAver+ testError.testError(numIterSort);            
            testErrorArray = cat(1,testErrorArray,testError.testError(numIterSort(alphaNum)));
          %  testError.testError(numIterSort)
          numIterSort(alphaNum)
          testError.testError(numIterSort(alphaNum))
          
  
% train error

            trainErrorName=strcat('train_error_',alphaName,'_', testName,'.mat');          
            trainError = load(trainErrorName);

           % trainErrorAver = trainErrorAver+trainError.trainError(numIterSort);
           % size(trainErrorAver);
            trainErrorArray = cat(1,trainErrorArray,trainError.trainError(numIterSort(alphaNum)));

         end 

           
           testErrorAver= mean(testErrorArray, "all");
           testErrorMax=  max(testErrorArray);
           testErrorMin=  min(testErrorArray);

           trainErrorAver= mean(trainErrorArray, "all");
           trainErrorMax= max(trainErrorArray);
           trainErrorMin= min(trainErrorArray);
          

            trainErrorAverArray(alphaNum)=trainErrorAver;
            trainErrorAverArrayMax(alphaNum)=trainErrorMax;
            trainErrorAverArrayMin(alphaNum)=trainErrorMin;
            
            testErrorAverArray(alphaNum)=testErrorAver;
            testErrorAverArrayMax(alphaNum)=testErrorMax;
            testErrorAverArrayMin(alphaNum)=testErrorMin;

end


f103=figure(1217)
            
 
%errorbar(1:length(alphaRange),trainErrorAverArray,trainErrorAverArrayMax,trainErrorAverArrayMin,'o-','LineWidth',3)
errorbar(1:4,log(trainErrorAverArray),log(trainErrorAverArray)-log(trainErrorAverArrayMin),log(trainErrorAverArrayMax)-log(trainErrorAverArray),'o-','LineWidth',3.5)

hold on
            

errorbar(1:4,log(testErrorAverArray),log(testErrorAverArray)-log(testErrorAverArrayMin),log(testErrorAverArrayMax)-log(testErrorAverArray),'x-','LineWidth',3.5)
            
         

hold on 

plot(1:4,log(k^(61/32)*rStar^(1/8)*(n-rStar)^(3/8)*(1.48)^21*(alpha).^(21/16)),'--','LineWidth',3.5)
            

legend('train-error', 'test-error', '$c\cdot\alpha^{21/16}$','interpreter','latex','fontsize',20);


xticks(1:4)
xticklabels({'10^{-2}','10^{-4}','10^{-6}', '10^{-8}'})
ax = gca;
ax.XAxis.FontSize = 14; % Change xticklabel font size
ax.YAxis.FontSize = 14; % Change yticklabel font size
xlabel('$\alpha$','FontSize', 30,'interpreter','latex');
hold off

