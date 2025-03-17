% Load extracted signals
wifiData = load('WIFI.mat');       
bluetoothData = load('bluetooth.mat'); 

% Extract actual waveform from struct
WIFI = wifiData.WIFI.waveform;  
bluetooth = bluetoothData.bluetooth.waveform;

% Ensure both signals have the same length
minLen = min(length(WIFI), length(bluetooth));
WIFI = WIFI(1:minLen);  
bluetooth = bluetooth(1:minLen);  

fs = 20e6;  % Sampling rate (Hz)
t = (0:minLen-1)'/fs;  % Correct time vector

% Frequency shift Bluetooth signal to avoid overlap
freqShift = 5e6;  
bluetoothShifted = bluetooth .* exp(1j*2*pi*freqShift*t);

% Combine Wi-Fi with shifted Bluetooth signal
combinedSignal = WIFI + bluetoothShifted;

% Save combined signal
save('combined_signal.mat', 'combinedSignal', 'fs');

disp("✅ Combined signal saved successfully!");
