//Start it
~rO_flute_interact_B_1.value;

//paramters
~maxAmp=1; ~minAmp=0.0002; ~ampCurve=1;
~maxPan= 2;
~stopValue= 1;
~threshold=0.05;
~minPitch=[48,48,24,24].midicps;
~maxPitch=[76,76,84,84].midicps;
~fbk1min=0.28; ~fbk1max=0.65; ~fbk2min=0.2; ~fbk2max=0.3; ~fbkCurve= 0;
~freqFbkFactorMin=1; ~freqFbkFactorMax=1.0; ~flowFbkFactor=1 ; ~enFbkFactor=0;
~freqFbkFactorMin=1; ~freqFbkFactorMax=1.1;
~minLpf=2000; ~maxLpf=5000;
~minBreath=0.005; ~maxBreath=0.1; ~breathCurve=0; ~flowBreathFactor= 1; ~enBreathFactor=0;
~maxVibDepth= 3.0; ~maxVibRate=8; ~minVibRate=8;
~dataMeanLengthFlow=6; ~dataMeanLengthEnergy=3;