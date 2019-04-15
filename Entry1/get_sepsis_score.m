function get_sepsis_score(input_file, output_file)
    
    % create all the necessary global variables
    
    global means
    global stds
    global fiss
    global KnnModel
    
    means.set1=[83.8996 97.0520 36.8055 126.2240 86.2907 66.2070 18.7280 33.7373];
    means.set2=[-3.1923   22.5352    0.4597  7.3889	39.5049   96.8883...
        103.4265   22.4952   87.5214    7.7210  106.1982	1.5961	0.6943...
        131.5327    2.0262    2.0509    3.5130  4.0541    1.3423    5.2734...
        32.1134   10.5383   38.9974   10.5585  286.5404  198.6777];
    means.set3=[60.8711 0.5435 0.0615 0.0727 -59.6769 28.4551];

    stds.set1=[17.6494 3.0163 0.6895 24.2988 16.6459 14.0771 4.7035 11.0158];
    stds.set2=[3.7845    3.1567    6.2684   0.0710    9.1087    3.3971...
        430.3638   19.0690   81.7152    2.3992    4.9761    2.0648    1.9926...
        45.4816    1.6008    0.3793     1.3092    0.5844    2.5511   20.4142...
        6.4362    2.2302   29.8928    7.0606  137.3886   96.8997];
    stds.set3=[16.1887 0.4981 0.7968 0.8029 160.8846 29.5367];
    
    
    fiss=sugfis;

    for ii=1:9
        fiss(ii)=readfis(['fis' num2str(ii)]);
    end
    
    load('KnnModel.mat','Mdl');
    KnnModel=Mdl;

    % generate scores
    data = read_challenge_data(input_file);

    % make predictions
    [scores, labels] = compute_sepsis_score(data);

    fid = fopen(output_file, 'wt');
    fprintf(fid, 'PredictedProbability|PredictedLabel\n');
    fclose(fid);
    dlmwrite(output_file, [scores labels], 'delimiter', '|', '-append');
end
        

function [scores, labels] = compute_sepsis_score(data)

global means
global sizes

sizes.TimeSample=size(data,1);
sizes.set1=size(means.set1,2);
sizes.set2=size(means.set2,2);
sizes.set3=size(means.set3,2);

[set1,set2,set3]=PrepareData(data);

% Evaluate Model for set1
out1=getOutputSet1(set1);

% Evaluate Model for set2
out2=getOutputSet2(set2);

% Evaluate Model for set3
out3=getOutputSet3(set3);

% Consider all feature sets 
scores=0.3*out1+0.05*out2+0.65*out3;

% Consider previous time steps
scores(2:end)=scores(2:end)+(scores(1:end-1)-0.6);
scores(3:end)=scores(3:end)+0.9*(scores(1:end-2)-0.6);
scores(4:end)=scores(4:end)+0.7*(scores(1:end-3)-0.6);
scores(scores>1)=1;

labels=round(scores);
end

function data = read_challenge_data(filename)
    f = fopen(filename, 'rt');
    try
        l = fgetl(f);
        column_names = strsplit(l, '|');
        data = dlmread(filename, '|', 1, 0);
    catch ex
        fclose(f);
        rethrow(ex);
    end
    fclose(f);

    % ignore SepsisLabel column if present
    if strcmp(column_names(end), 'SepsisLabel')
        column_names = column_names(1:end-1);
        label= data(:,end);
        data = data(:,1:end-1);
    end
end

function [set1,set2,set3]=PrepareData(data)

global sizes
global means

set1=data(:,1:8);
set2=data(:,9:34);
set3=data(:,35:40);

for k=1:sizes.set1, set1(isnan(set1(:,k)),k)=means.set1(k);end
for k=1:sizes.set2, set2(isnan(set2(:,k)),k)=0;end
for k=1:sizes.set3, set3(isnan(set3(:,k)),k)=0;end

f1=set1(:,1);
f2=set1(:,2);
f3=set1(:,3);
f4=set1(:,4);
f5=set1(:,5);
f6=set1(:,6);
f7=set1(:,7);
f8=set1(:,8);

f1=intensity_stretching(f1,[80 100],'Type1');
f2=intensity_stretching(f2,2,'Type3');              % Reduce the effect X2
f3=intensity_stretching(f3,[37 1]  ,'Type4');       % Gaussian with mean=37,std=1
f4=intensity_stretching(f4,2,'Type3');
f5=intensity_stretching(f5,[70 90] ,'Type1');
f6=intensity_stretching(f6,[50 80] ,'Type1');
f7=intensity_stretching(f7,[20 25] ,'Type5');
f8=intensity_stretching(f8,[25 35] ,'Type5');       % Soft sigmoid between 25 and 35

set1=[f1 f2 f3 f4 f5 f6 f7 f8];

end

function [ out ] = intensity_stretching( x,params,type)


maxIn=max(max(x));
minIn=min(min(x));
maxY=maxIn;


switch type
    case 'Type1'
        xbp=[minIn params(1) params(2) maxIn];
        ybp=maxY*[0 0.3 0.7 1];
        out=intensity_stretching_linear(x,xbp,ybp);
    case 'Type2'
        xbp=[minIn params(1) params(2) maxIn];
        ybp=maxY*[0 0.4 0.6 1];
        out=intensity_stretching_linear(x,xbp,ybp);
    case 'Type3'
        if params==0
            a=maxIn;
        else
            a=params(1);
        end
        out=x./a;
    case 'Type4'
        mea=params(1);
        std=params(2);
        out=maxY*gaussmf(x,[std mea]);
    case 'Type5'
        a=params(1);
        b=params(2);
        out=maxY*smf(x,[a b]);
end


end

function [out] = intensity_stretching_linear(input,xbp,ybp)
%INTENSITY_STRETCHING_LINEAR Summary of this function goes here
%   Detailed explanation goes here

m1=(ybp(2)-ybp(1))/(xbp(2)-xbp(1));
m2=(ybp(3)-ybp(2))/(xbp(3)-xbp(2));
m3=(ybp(4)-ybp(3))/(xbp(4)-xbp(3));

[s1,s2]=size(input);
out=zeros(s1,s2);

for ii=1:s1
    for jj=1:s2
        p=input(ii,jj);
        if p<=xbp(1)
            % Out of bounds
            a=ybp(1);
        elseif p<=xbp(2)
            % In Line 1
            a=round(m1*(p-xbp(1))+ybp(1));
        elseif p<=xbp(3)
            % In Line 2
            a=round(m2*(p-xbp(2))+ybp(2));
        elseif p<=xbp(4)
            % In Line 3
            a=round(m3*(p-xbp(3))+ybp(3));
        else
            % Out of bounds
            a=ybp(4);
        end
        out(ii,jj)=a;
    end
end
end

function [out]=getOutputSet1(set1)

global fiss


l=zeros(size(set1));

for ii=1:8, l(:,ii)=evalfis(fiss(ii),set1(:,ii));end

out=evalfis(fiss(9),l);

end

function [out]=getOutputSet2(set2)
    out=zeros(size(set2,1),1);
    Influence=0.1*[10 8 8 3 2 7 8 2 8 2 2 2 2 2 2 2 2 2 2 2 2 2 1 10 1 1];
    upLimit  =[5 30 1 7.6 70 100 4000 100 200 10 120 5 7 500 15 4 7 7 5 50 50 15 60 100 400 400];
    downLimit=[-5 15 0 7 20 80 0 0 0 0 100 0 0 100 0 1 1 2 0 0 20 5 10 0 0 0];
    
    for ii=1:size(set2,2)
        d=set2(:,ii);
        a=zeros(size(out));
        a(d>downLimit(ii) & d<upLimit(ii))=1;
        out=out+a*Influence(ii);
    end
end

function [out]=getOutputSet3(set3)
global KnnModel
out=predict(KnnModel,set3);
end