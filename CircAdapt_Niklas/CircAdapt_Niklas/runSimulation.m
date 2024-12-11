function [Part, flow, cav, diff] = runSimulation(C, R, S, DtSimulation)

load('P');

global P; % P contains all parameters of the CVS needed for the simulation and simulation results

%disp(['Current sample: [', num2str(sample(1)), ', ', num2str(sample(2)), ', ', num2str(sample(3)), ']']);

%% Do the computation
G=P.General;
G.DtSimulation=DtSimulation; % 9s simulation duration = about 10 cycles to steady state
G.PressFlowContr=0;
disp(G.PressFlowContr)

%% set parameter space values (6-4-10 equals to 1-1-1 factors thus no change from homeostatic values)
% Contractility of the whole heart.
SfActRef = GetFt('Patch', 'SfAct', {'La1', 'Ra1', 'Lv1', 'Rv1'});
PutFt('Patch', 'SfAct', {'La1', 'Ra1', 'Lv1', 'Rv1'}, SfActRef*C)

% Systemic resistance as in Adapt0P
p0AVRef = GetFt('ArtVen','p0AV','Sy');
PutFt('ArtVen','p0AV','Sy', p0AVRef*R)

% Venous compliance (2,1) in k
kRef = GetFt('ArtVen','k','Sy');
PutFt('ArtVen','k','Sy', [kRef(1); kRef(2)*S]);

%% run CircAdapt
G.tEnd=P.t(end)+G.DtSimulation;
P.General=G;

CircAdaptP;
    


%% Post-process results
% Access solution for arterial pressure and volumes of left ventricle
p1=GetFt('Node','p',{'SyArt','Lv','La'})/133; % in mmHg

Part=p1(:,1);
first = Part(1); last = Part(end);
diff = abs(last-first);

flow1 = P.Valve.q(:,1);
flow2 = P.Valve.q(:,2);
flow3 = P.Valve.q(:,3);
flow4 = P.Valve.q(:,4);
flow5 = P.Valve.q(:,5);
flow6 = P.Valve.q(:,6);
flow7 = P.Valve.q(:,7);
flow8 = P.Valve.q(:,8);

flow = [flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8];

cav1 = P.Cavity.V(:,1);
cav2 = P.Cavity.V(:,2);
cav3 = P.Cavity.V(:,3);
cav4 = P.Cavity.V(:,4);
cav5 = P.Cavity.V(:,5);
cav6 = P.Cavity.V(:,6);
cav7 = P.Cavity.V(:,7);
cav8 = P.Cavity.V(:,8);

cav = [cav1, cav2, cav3, cav4, cav5, cav6, cav7, cav8];

outputFileName = sprintf('CircAdapt_Niklas\CircAdapt_Niklas/p_structures/P_%d_%d_%d.mat', round(C, 2), round(R, 2), round(S, 2));
save(outputFileName, "P")

end