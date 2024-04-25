function PStructure = loadWorkspace(sample)

% Clear global workspace and load new P structure
clear global

fileName = char('P_C' + string(sample(1)) + '_E' + string(sample(2)) + '_R' ...
    + string(sample(3)) + '.mat');

PStructure = load(fileName);

fprintf('%s loaded! \n',fileName);