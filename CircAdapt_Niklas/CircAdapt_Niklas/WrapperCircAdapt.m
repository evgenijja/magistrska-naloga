%% Test wrapper function for calling main function
clear global

% sampleDefault = [0.1; 1.0; 1.0]; 
sample = [1.0; 1.0; 1.5]; % Csvn, Emaxlv, Rsart

% [MAP; PP; CO]
outputs = CircAdapt4CompilerCall(sample);