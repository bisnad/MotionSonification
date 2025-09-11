
~Voice_Synthesis_4sensors.value
//1. VOICES
~minFreq=[48,48,25,24].midicps;
~minFreq=[48,48,35,35].midicps;
~maxFreq=[84,84,84,84].midicps;
~minFreq=[12,12,12,12].midicps;  //viejo
~maxFreq=[76,76,84,84].midicps;//viejo

~maxVibDepth= 4; ~maxVibRateSoprano=9; ~minVibRateSoprano=6;
~dataMeanLengthFlow=2;

~minLpf= 12000;
~maxLpf= 21000;
~lowVoiceA= 'bassA';~lowVoiceE= 'bassE';~lowVoiceI= 'bassI'; ~lowVoiceO= 'bassO'; ~lowVoiceU= 'bassU';
~lowVoiceA= 'tenorA';~lowVoiceE= 'tenorE';~lowVoiceI= 'tenorI'; ~lowVoiceO= 'tenorO'; ~lowVoiceU= 'tenorU';
~threshold=0.035;
~stopValue= 1;
~specMaxAmp=0.3;
~specMinAmp=0.002;
~maxPan= 0.75 // for 7 channels
~stopValue= 1;
~threshold=0.05;
~minFreq=24.midicps;
~maxFreq=84.midicps

~lowVoiceA= 'tenorA';~lowVoiceE= 'tenorE';~lowVoiceI= 'tenorI'; ~lowVoiceO= 'tenorO'; ~lowVoiceU= 'tenorU';
~highVoiceA= 'sopranoA';~highVoiceE= 'sopranoE';~highVoiceI= 'sopranoI';~highVoiceO= 'sopranoO';~highVoiceU= 'sopranoU';

//stop it


4.do{|x|~respondersRaw[x].disable}; 4.do{|x|~respondersFlow[x].disable}; 8.do{|x|~synths[x].free}; 8.do{|x|~reverb[x].free};


