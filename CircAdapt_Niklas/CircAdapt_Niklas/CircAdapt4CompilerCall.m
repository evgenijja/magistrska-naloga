function outputs = CircAdapt4CompilerCall(sample)

% clear global

%% Load workspace for seed point of sample
% fileName = 'P_C' + string(sampleSeed(1)) + '_E' + string(sampleSeed(2)) + '_R' ...
%     + string(sampleSeed(3)) + '.mat';
% load(fileName);
% fprintf('Workspace loaded: %s\n',fileName);

load('P');

global P; % P contains all parameters of the CVS needed for the simulation and simulation results

disp(['Current sample: [', num2str(sample(1)), ', ', num2str(sample(2)), ', ', num2str(sample(3)), ']']);

%% Do the computation
G=P.General;
G.DtSimulation=34; % 9s simulation duration = about 10 cycles to steady state

% parameter space values
Vk=sample(1); % Venous compliance: [0.35 0.40 0.45 0.55 0.60 0.65 0.70 0.75 0.85 1 3]
Vi=sample(2); % LV contractility: [0.35 0.45 0.55 0.7 0.8 1 1.2 1.4 1.6 1.8 2]
Vj=sample(3); % R peripheral: [2.5 2 1.5 1 0.8 0.6 0.3 0.16 0.08 0.04 0.02]

%% set parameter space values (6-4-10 equals to 1-1-1 factors thus no change from homeostatic values)
% Contractility of the whole heart.
SfActRef = GetFt('Patch', 'SfAct', {'La1', 'Ra1', 'Lv1', 'Rv1'});
PutFt('Patch', 'SfAct', {'La1', 'Ra1', 'Lv1', 'Rv1'}, SfActRef*Vi)

% Systemic resistance as in Adapt0P
p0AVRef = GetFt('ArtVen','p0AV','Sy');
PutFt('ArtVen','p0AV','Sy', p0AVRef*Vj)

% Venous compliance (2,1) in k
kRef = GetFt('ArtVen','k','Sy');
PutFt('ArtVen','k','Sy', [kRef(1); kRef(2)*Vk]);

%% run CircAdapt
G.tEnd=P.t(end)+G.DtSimulation;
P.General=G;
CircAdaptP; % generate solution

%% Post-process results
% Access solution for arterial pressure and volumes of left ventricle
p1=GetFt('Node','p',{'SyArt','Lv','La'})/133; % in mmHg
Part=p1(:,1);

DP = min(Part);
SP = max(Part);
PP = SP-DP;
MAP = DP + 1/3*(PP);

CalV= 1e6; % m^3 to cm^3
Vlv = CalV*GetFt('Cavity','V',{'Lv'}); % mL
EDV = max(Vlv);
ESV = min(Vlv);
HR = (P.General.tCycle)^-1;
CO = (EDV-ESV)*HR*(60/1000); % L/min

outputs = [MAP; PP; CO];