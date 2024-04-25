function CircAdapt4CompilerCall;
% Version 2, 27.11.2023 
% new parameter kAV-> p0AV
% apdated ranges

clear
global P; %P contains all parameters of the CVS needed for the simulation and simulation results

load('P.mat'); %
G=P.General;
G.DtSimulation=9; % 9s simulation duration = about 10 cycles to steady state

%parameter space values
Vi=[0.55 0.65 0.75 0.85 0.9 1 1.2 1.4 1.6 1.8 2];% LV contractility
Vj=[0.7 0.76 0.82 0.88 0.94 1 1.1 1.2 1.3 1.4 1.5];% R systemic resistence
Vk=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 3];% venous stiffnes (1/compliance)

%set parameter space values (6-4-10 equals to 1-1-1 factors thus no change from homeostatic values)
P.Patch.SfAct(1:5)=  P.Patch.SfAct(1:5)*Vi(6);% set LV contractility
P.ArtVen.p0AV(1,1)=  P.ArtVen.p0AV(1,1)*Vj(6);% set R systemic resistence
P.ArtVen.k(2,1)=     P.ArtVen.k(2,1)*Vk(10);% set venous stifnes (compliance)

%run CircAdapt
G.tEnd=P.t(end)+G.DtSimulation;
P.General=G;
CircAdaptP; %generate solution

%access the solution
p1=GetFt('Node','p',{'SyArt','Lv','La'})/133;% in mmHg
Part=p1(:,1);
plot(Part)