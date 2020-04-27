%reload_ext autoreload
%autoreload 2
%matplotlib inline

data = importdata('u.data', '\t');

norm_data = centerData(data);   %% kentrarisma


[number_of_reviews,users] = groupcounts(norm_data(:,1)); 
input = zeros(length(users),20);
input(:,1) = users;

for i = 1:length(norm_data)
    input(norm_data(i,1),norm_data(i,2)+1) = norm_data(i,3); 
end

%user = input(:,1);

data = scaledata(input, 0, 1);    %%kanonikopoihsi


categories_c = num2cell(input(:,1));

c = cvpartition (input(:,1), 'KFold', 5, 'Stratify', false);


u = 0 + 1*rand(943,6);
m = 0 + 1*rand(6,1683);

%i=input(:,2:1683);

C = u*m;

%y1 = myNeuralNetworkFunction(input);

net = BP_TB(C,input,0.01,0.01,1,'yes');
out = predict(net,input);


%%----- Synartisi gia centering dedomenwn

function cent_data = centerData(data)
    [number_of_reviews, user] = groupcounts(data(:,1)); 
    user_average = zeros(length(user), 2);
    user_average(:,1) = user;
    cent_data = data;
    
    for i = 1:length(data(:,1))
        current_user = data(i,1);
        current_review = data(i,3);
        user_average(current_user,2) = user_average(current_user,2) + current_review;
    end
    
    user_average(:,2) = user_average(:,2)./number_of_reviews(:);

    for i = 1:length(cent_data)
        current_user = cent_data(i,1);
        cent_data(i,3) = cent_data(i,3) - user_average(current_user,2);
    end
end




%%----- Synartisi kanonikopoihsis dedomenwn
function normData = scaledata(data,minval,maxval)
    normData = data - min(data(:));
    normData = (normData/range(normData(:)))*(maxval-minval);
    normData = normData + minval;
end