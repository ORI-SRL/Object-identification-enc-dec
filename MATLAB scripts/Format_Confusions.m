%% Load previous object predictions and then plot confusion matrices

all_data = table2array(readtable('../saved_model_states/iterative/IterativeRNN_labels.csv',...
            'ReadVariableNames', false));
% extract from square brackets
split_data = split(all_data,["[","]"]);
split_data(:, :, [1,3]) = [];
for g = 1:size(split_data,2)
    grasp_data = split(split_data(:, g),",");
    cm1 = plotconfusion(grasp_data(1,:), grasp_data(2,:)); %confusionmat(
    %cm1 = cm1./sum(cm1,2);
    figure;
    cm2 = confusionchart(cm1);
    
    
end



hidden = [495 652 711 786 853 895 910 926 940 955];
prediction = [ 470 625 741 803 850 884 921 941 947 956];
plot(a); hold on; plot(b)
legend('Recursive Prediction','Recursive Hidden')
