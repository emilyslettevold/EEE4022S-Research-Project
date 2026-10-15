%% === Step 1: Load raw IMU CSVs ===
acc = readtable('acc_data_final.csv');
gyr = readtable('gyr_data_final.csv');
disp("Read tables");

%% === Step 2: Keep IMU1 / pre-filter rows only ===
% Gyro: I=0 is the pre-filter (raw) stream, I=1 is post-filter.
% Accel: only I=0 exists.
if ismember('I', gyr.Properties.VariableNames)
    gyr = gyr(gyr.I == 0, :);
end
if ismember('I', acc.Properties.VariableNames)
    acc = acc(acc.I == 0, :);
end

%% === Step 3: Timing check - gyro (SampleUS = capture time) ===
gyrTime = gyr.SampleUS / 1e6;
gaps = diff(gyrTime);

figure; plot(gaps);
xlabel('Sample index'); ylabel('Time gap to next sample (s)');
title('Gyro inter-sample timing check (using SampleUS)');

fprintf('--- Gyro timing check ---\n');
fprintf('Mean gap: %.8f s\n', mean(gaps));
fprintf('Std dev of gaps: %.8f s\n', std(gaps));
fprintf('Max gap: %.8f s\n', max(gaps));
fprintf('Min gap: %.8f s\n', min(gaps));

%% === Step 4: Gyro vectors ===
gyroFs = 1 / mean(gaps);
gyro_x = gyr.GyrX';
gyro_y = gyr.GyrY';
gyro_z = gyr.GyrZ';

%% === Step 5: Timing check + vectors - accelerometer ===
accTime = acc.SampleUS / 1e6;
accGaps = diff(accTime);

fprintf('--- Accel timing check ---\n');
fprintf('Mean gap: %.8f s\n', mean(accGaps));
fprintf('Std dev of gaps: %.8f s\n', std(accGaps));
fprintf('Max gap: %.8f s\n', max(accGaps));
fprintf('Min gap: %.8f s\n', min(accGaps));

accelFs = 1 / mean(accGaps);
accel_x = acc.AccX';
accel_y = acc.AccY';
accel_z = acc.AccZ';

%% === Step 6: Allan variance, all 6 axes ===
[avar_gx, tau_gx] = allanvar(gyro_x,  'octave', gyroFs);  adev_gx = sqrt(avar_gx);
[avar_gy, tau_gy] = allanvar(gyro_y,  'octave', gyroFs);  adev_gy = sqrt(avar_gy);
[avar_gz, tau_gz] = allanvar(gyro_z,  'octave', gyroFs);  adev_gz = sqrt(avar_gz);
[avar_ax, tau_ax] = allanvar(accel_x, 'octave', accelFs); adev_ax = sqrt(avar_ax);
[avar_ay, tau_ay] = allanvar(accel_y, 'octave', accelFs); adev_ay = sqrt(avar_ay);
[avar_az, tau_az] = allanvar(accel_z, 'octave', accelFs); adev_az = sqrt(avar_az);

%% === Step 7: Plots ===
figure;
loglog(tau_gx, adev_gx, tau_gy, adev_gy, tau_gz, adev_gz);
grid on; legend('Gyro X','Gyro Y','Gyro Z');
xlabel('\tau (s)'); ylabel('\sigma(\tau)'); title('Gyroscope Allan Deviation');

figure;
loglog(tau_ax, adev_ax, tau_ay, adev_ay, tau_az, adev_az);
grid on; legend('Accel X','Accel Y','Accel Z');
xlabel('\tau (s)'); ylabel('\sigma(\tau)'); title('Accelerometer Allan Deviation');

%% === Step 8: Kalibr parameters ===
% N: raw curve value at tau = 1 s
% K: fitted +1/2 slope line, read at tau = 3 s
axesData = struct( ...
    'gyro_x',  struct('tau', tau_gx, 'adev', adev_gx), ...
    'gyro_y',  struct('tau', tau_gy, 'adev', adev_gy), ...
    'gyro_z',  struct('tau', tau_gz, 'adev', adev_gz), ...
    'accel_x', struct('tau', tau_ax, 'adev', adev_ax), ...
    'accel_y', struct('tau', tau_ay, 'adev', adev_ay), ...
    'accel_z', struct('tau', tau_az, 'adev', adev_az));

axisNames = fieldnames(axesData);
N_vals = zeros(numel(axisNames), 1);
K_vals = zeros(numel(axisNames), 1);

for k = 1:numel(axisNames)
    tau  = axesData.(axisNames{k}).tau;
    adev = axesData.(axisNames{k}).adev;
    logtau   = log10(tau);
    logadev  = log10(adev);
    dlogadev = diff(logadev) ./ diff(logtau);

    [~, idx1] = min(abs(tau - 1));
    N_vals(k) = adev(idx1);

    slope = 0.5;
    [~, i] = min(abs(dlogadev - slope));
    b = logadev(i) - slope*logtau(i);
    K_vals(k) = 10^(slope*log10(3) + b);
end

%% === Step 9: Per-axis table ===
resultsTable = table(axisNames, N_vals, K_vals, ...
    'VariableNames', {'Axis', 'N_NoiseDensity', 'K_RandomWalk'});
disp('=== Per-axis noise parameters ===');
disp(resultsTable);

%% === Step 10: Averages -> imu.yaml values ===
gyro_idx  = contains(axisNames, 'gyro');
accel_idx = contains(axisNames, 'accel');

gyroscope_noise_density     = mean(N_vals(gyro_idx));
gyroscope_random_walk       = mean(K_vals(gyro_idx));
accelerometer_noise_density = mean(N_vals(accel_idx));
accelerometer_random_walk   = mean(K_vals(accel_idx));

fprintf('\n=== Kalibr imu.yaml values ===\n');
fprintf('gyroscope_noise_density: %.10e\n', gyroscope_noise_density);
fprintf('gyroscope_random_walk: %.10e\n', gyroscope_random_walk);
fprintf('accelerometer_noise_density: %.10e\n', accelerometer_noise_density);
fprintf('accelerometer_random_walk: %.10e\n', accelerometer_random_walk);