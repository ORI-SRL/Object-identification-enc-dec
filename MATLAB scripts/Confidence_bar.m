%% present a bar chart that displays when each object has reached a 
%   particular confidence threshold
close all; clear
Objects = categorical({'apple', 'bottle', 'cards', 'cube', 'cup', 'cylinder', 'sponge'});

conf = [4, 4, 7, 10, 4, 4, 4];
%stdev at point of reaching >85% accuracy
app_dev = std([0.0, 2.1, 1.4, 2.9, 7.1, 0.0]);
bott_dev = std([4.6, 0.0, 0.0, 0.6, 2.3, 0.0]);
card_dev = std([1.4, 0.0, 4.2, 0.7, 0.0, 4.2]);
cube_dev = std([0.0, 0.0, 8.3, 6.1, 0.0, 0.0]);
cup_dev = std([0.0, 0.0, 5.5, 2.7, 4.1, 0.0]);
cyl_dev = std([0.8, 4.1, 0.0, 0.0, 0.0, 0.0]);
spon_dev = std([3.9, 0.0, 7.8, 0.0, 2.6, 0.0]);

%accuracies over time
time_acc = [];
time_acc(1,:) = [0 75.0 77.9   84.1 86.4 90.9 91.8 97.2    96.9   99.4    100];  %app_acc
time_acc(2,:) = [0 44.6 69.9   83.1 92.5 97.3 99.0 100     100    99.4    100];  %boot_acc 
time_acc(3,:) = [0 43.5 46     56.1 63.3 79.3 80.8 89.4    88.2   96.9    96.2]; %card_acc 
time_acc(4,:) = [0 33.1 44.4   48.5 54.7 62.4 65.4 81.8    80     84.1    85.6]; %cube_acc 
time_acc(5,:) = [0 57.3 65.7   83.7 87.7 93.5 96.1 92.3    96.9   97.9    100];  %cup_acc 
time_acc(6,:) = [0 54.8 66.9   80.4 95.0 96.7 96   99.3    100    99.4    100];  %cyl_acc 
time_acc(7,:) = [0 38.8 72.1   80.3 85.7 90.6 97.7 98.7    99.3   100     100];  %spon_acc 
t_acc_mean = mean(time_acc,1);
err = [app_dev, bott_dev, card_dev, cube_dev, cup_dev, cyl_dev, spon_dev];

obj_diffs = diff(time_acc, [], 2);
grasp_diffs = mean(obj_diffs);

discrete_idxs = (1:10)+1;
for r = 1:size(time_acc,1)
    disc_bars(r,:) = [time_acc(r, discrete_idxs)];
end
% plot bar charts 
figure
d = boxplot(obj_diffs); ax = gca;
%boxplot(time_acc(:,2:end)', Objects); ax = gca; ax.YLim = [0, 105];
figure
hold on
for r = 1:size(disc_bars,1)
    b = bar(Objects(r), disc_bars(r,:)); %err, 'vertical', 'b.'
    if r == 1
        for c = 1:size(b(:),1)
            Colour_pool(c,:) = get(b(c),'FaceColor');
        end
    end
    for c = 1:size(b(:),1)
        set(b(c), 'FaceColor', Colour_pool(r,:))
    end
        
end
figure
m = bar(Objects, max(time_acc, [], 2)); 

figure
hold on
for r = 1:size(time_acc,1)
    plot(0:10,time_acc(r,:))
end
plot(0:10, t_acc_mean, 'LineWidth',2)
ax = gca;
ax.YLim = [0,101];
ax.XLabel.String = 'Number of grasps';
ax.YLabel.String = 'Classification Accuracy / %';
legend(Objects)
ax.Legend.Location = "southeast";