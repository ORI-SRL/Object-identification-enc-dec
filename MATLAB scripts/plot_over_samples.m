%% Check the input data and plot the sensor outputs over the entire test
% close all
clear
filepath = "C:\Users\shil4283\OneDrive - Nexus365\DPhil Work\Python\DataShuffling\combined_old_new\";
%"C:\Users\shil4283\OneDrive - Nexus365\DPhil Work\Python\DataShuffling\data_11_03_22\"; %'D:\Object_reco\tuning_exps\tune_cards\skin_data\';
objects = {'apple'; 'bottle'; 'cards'; 'cube'; 'cup'; 'cylinder'; 'sponge'};
old_count = zeros(1,19); new_count = zeros(1,19);
fig1 = figure; hold on; 
% fig2 = figure; hold on;
% fig.Title.String = objects(obj);
for obj = 1:size(objects)
    obj_folder = strcat(objects{obj}, '_all');
% [file_in, filepath] = uigetfile('*');
    all_obj_files = dir(strcat(filepath, obj_folder));
    
    for file = 1:size(all_obj_files)-2
        filename =  all_obj_files(file+2).name;
        txt_in = dlmread(strcat(filepath, obj_folder, "\", filename));
%         end_grasp_time = txt_in(end, 2) - 5.72;
%         [~,end_idx] = min(abs(txt_in(:, 2) - end_grasp_time));
%         dlmwrite(strcat(filepath, obj_folder, "\", all_obj_files(file+2).name),... %"..\data_11_03_22_shifted\", 
%             txt_in(1:end_idx, :), 'delimiter', ' ');
        init_row = txt_in(6,3:end);
        end_row = txt_in(end,3:end);
        delta = end_row - init_row;
        delta(delta<1) = 0;
        if obj<5
            subplot(2,4,obj)
            
        else
            subplot(2,3,obj-1)
            
        end
        hold on
        x = linspace(1,19,19);
        test_num = str2double(filename(11:end-2));
        if mod(test_num, 2) == 0
            new_count = new_count + delta;
%             delta = [delta(1:16), delta(19), delta(17:18)];
            plot(x, delta, 'r')
        else
            old_count = old_count + delta;
            plot(x, delta, 'b')
        end
        fig = gca;
        fig.Title.String = objects(obj);
        fig.XLabel.String = "sensor number";
        fig.YLim = [0, 100];
    end
end

scaling = 200*7;
old_means = old_count/scaling;
new_means = new_count/scaling;
% new_count_switch = [new_count(1:16), new_count(19), new_count(17), new_count(18)];
figure; hold on;
plot(old_count)
plot(new_count)
plot(new_count_switch)
legend(["Old data", "New data", "Switched new"])
%%
% rel_data = txt_in(6:end,3:end);
% time = txt_in(6:end, 2) - txt_in(6, 2);
% 
% figure
% hold on
% for k = 1:size(rel_data, 2)
%     plot(rel_data(:,k))
% end
% 
% diffs = diff(rel_data);
% figure
% hold on
% for k = 1:size(rel_data, 2)
%     plot(time(2:end), diffs(:,k))
% end
% 
% output_del = rel_data(end,:) - rel_data(1, :);
% disp(output_del)