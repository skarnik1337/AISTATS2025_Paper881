clear all 
%close all
clc

tic

addpath('../')

%% all test params
%n= 10;
n= 10;
k= 4;
rStar=3;

numData=ceil(0.9*(n+1)*rStar*k*log(n+1));

alphaRange= [2 4 6 8];      % initialization scale 
testNumberRange=1:20;
numIter= 4000;
r=200;                      % overparam

testNumRangeName=["test_1", "test_2","test_3", "test_4", "test_5", "test_6", "test_7","test_8", "test_9", "test_10","test_11", "test_12","test_13", "test_14", "test_15"...
    "test_16", "test_17","test_18", "test_19", "test_20"]; 
alphaRangeName= ["minus_2", "minus_4","minus_6","minus_8"]; 

%numIter= 500;

PowerMType='sym';
%PowerMType='non-sym';

% setting for normalized tensors
TesorType='norm'; 
nu= 1e-5;       % learning rate 

% % % setting NON for normalized tensors 
% TesorType='non-norm'; 
% nu= 1e-6;       % learning rate 
% alpha= 1e-5;   % initialization


%% data
UStar=randn(n,rStar,k);

%% tensor type
switch TesorType

    case  'norm' 

    % normalized tensor
    [UUStar,SUStar,VUStar] = tSVD(UStar,'econ');
    XStar=tProduct(UUStar, tTranspose(UUStar));     % normalized tensor
    [~,SStar,~] = tSVD(UUStar, 'econ');
    SStar(:,:,1);

    case  'non-norm'

    [UUStar,~,VUStar] = tSVD(UStar,'econ');
    XStar=tProduct(UStar, tTranspose(UStar));      % non- normalized tensor
    [~,SStar,~] = tSVD(UStar, 'econ');
    [~,SStarX,~] = tSVD(XStar, 'econ');

end 
%%
GTest=randn(n, n, k, numData); 

G=zeros(n, n, k, numData);

for ii = 1:numData
    G(:,:,:,ii) = GTest(:,:,:,ii)+tTranspose(GTest(:,:,:,ii));
end

% data
y= zeros(numData,1); 

    for j = 1:numData
        y(j)=tensorprod(XStar, G(:,:,:,j),'all');
    end

    %% operators

IdTensor=zeros(n,n,k);
IdTensor(:,:,1)=eye(n);

% A^*A(UStar*UStar^T)
operatorStarToData= zeros(n,n,k); 

    for j = 1:numData
    
          operatorStarToData =  operatorStarToData + y(j)*G(:,:,:,j);
    end

%% PowerMType
switch PowerMType

    case 'non-sym'
    %  % NO symmetrization of the measurement operator 
    [L,SL,~] = tSVD(operatorStarToData,'econ'); % , 'econ'

    case 'sym'
    %   %  % symmetrization of the measurement operator 
    operatorStarToDataSym= 1/2*(operatorStarToData+ tTranspose(operatorStarToData));
    [L,SL,~] = tSVD(operatorStarToDataSym,'econ'); % , 'econ'

end 


%% set up for gradient 


normOfXStar=sqrt(tensorprod(XStar,XStar, 'all'));
normOfData=norm(y);
normSStar=norm(squeeze(SStar(rStar,rStar,:)));


%% gradient iterations 

for alphaNum=1:length(alphaRange)

    alpha=10^(-alphaRange(alphaNum));
    alphaName=alphaRangeName(alphaNum);

    U= alpha*randn(n,r,k)*(1/sqrt(r));
    U0=U;
    X=tProduct(U, tTranspose(U));

    for test=testNumberRange

        testName=testNumRangeName(test);

        clear  testError trainError RelTubalSingValErr PrincipalAngle
        
        count=0;
        
        testError=zeros(numIter,0);
        trainError = zeros(numIter,0);
        RelTubalSingValErr = zeros(numIter,1);
        
        
        %set up power method iterates 
        PrincipalAngle=zeros(numIter,1);
        PrincipalAngleToGroundTruth=zeros(numIter,1);
        PrincipalAngleTilde=zeros(numIter,1);

          for i= 1:numIter        
            
             count= count + 1;
            
             %%%%%%%   gradient   %%%%%%% 
            
               clear X newData gradUVal S
            
                newData=zeros(numData,1); 
            
                X=tProduct(U, tTranspose(U));
            
                testError(i) = sqrt(tensorprod(XStar-X, XStar-X,'all'))/normOfXStar;
            
                    if testError(i)>20 %isnan(testError(i))
                               disp('Error is NaN (Not a Number)          ----  !!!!');
                        break
                    end
            
                    for j = 1:numData
                         newData(j)=tensorprod(X, G(:,:,:,j),'all');
                    end 
            
                dataDiff=y-newData;
                trainError(i)= norm(dataDiff)/normOfData;
            
                gradUVal=gradUNew(U,G,y); 
                U_t= U-nu*gradUVal;           
                U=U_t;    
                [~,S,~] = tSVD(U, 'econ'); %,'econ'
               
                RelTubalSingValErr(i) = norm(squeeze(S(rStar,rStar,:)-SStar(rStar,rStar,:)))/normSStar;
                         
                 %%%%%%%   "power method"  %%%%%%% 

                 [V,~,~] = tSVD(X,'econ');
                 V_t=V(:,1:rStar,:);
            
                 [PrincipalAngle(i), PrincipalAngleAll]=tubal_principal_angle_Fourier_pages(V_t,LStar);
                         
        
          end 

    
            testErrorName=strcat('test_error_',alphaName,'_', testName,'.mat');
            trainErrorName=strcat('train_error_',alphaName,'_', testName,'.mat');
            principalAnglesName=strcat('principal_angle_error_', alphaName,'_', testName,'.mat');  
            singularTubesName=strcat('singular_tubes_', 'non',alphaName,'_', testName,'.mat');  
    
            save(testErrorName,'testError');%,"-ascii"
            save(trainErrorName,'trainError');
            save(principalAnglesName, 'PrincipalAngle');   
            save(singularTubesName, 'RelTubalSingValErr');  
            %a=load('train_error_over_50_test_2.mat')
     end 
  
end 


toc

%% new grad

function gradU =gradUNew(U, G, yTrue)

  %  clear operatorConj gradUVfnew gradU
    
    X= tProduct(U, tTranspose(U));
    [n1, n2, n3, numData]=size(G);

    operatorConj = zeros(n1,n2,n3); 

    for j = 1:numData

         operatorConj = operatorConj + (tensorprod(X, G(:,:,:,j),'all')-yTrue(j))*(G(:,:,:,j)+tTranspose(G(:,:,:,j)));
    end

    gradU=tProduct(operatorConj,U);

end 
