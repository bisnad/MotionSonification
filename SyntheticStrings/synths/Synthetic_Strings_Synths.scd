
//smooth strings with no reverb incorporated
SynthDef(\em_smooth_strings_noVerb, {arg freqRate=1, freqMul=200, freq=300,  lowCutfreq=22000, highCutfreq=10, gate=1, rq=1, decay=1, combDecay= 0.15, attack=0.01,amp=1.0,lagAmp=0, rel=0.3, pos=0, curve= -3, out=0, harmAmp=1, combAmp=0.5, roomSize=3, revtime=2.6, damping=0.2, inputbw=0.48, spread=15, drylevel= -3, earlylevel= -4, taillevel= -13, rqInit=0.005, rqEnd=0.0005, rqTime=0.1, freqFactor=0.02;
	 var  noise, demand, filt, env, freqLo, freqHi, pan, comb, verb,rqEnv, freqLFO;
	 freqLFO=LFDNoise3.kr(freqRate,freqMul,freq);
	freqLo= (freqLFO) - (freqLFO*freqFactor);
	freqHi= (freqLFO) + (freqLFO*freqFactor);
	 noise = PinkNoise.ar(2.0);
	 rqEnv = EnvGen.ar(Env([rqInit, rqEnd],[rqTime]), doneAction:0);
	 env = EnvGen.ar(Env.asr(attack, 1, decay,curve ), gate, doneAction:2);
	 demand = Duty.ar(0.01, 0,Dbrown(freqLo, freqHi,1));
	 filt   = (1..30).collect{|harmonic|BPF.ar(noise, demand*harmonic, rqEnv*rq, (1/(harmonic**harmAmp)))};
     filt   = Mix(LPF.ar(filt, lowCutfreq));
     filt   = HPF.ar(filt,highCutfreq);
     comb   = CombN.ar(filt, 0.2, freqLFO.reciprocal, combDecay, combAmp );
    // verb   =  GVerb.ar(filt +comb, roomSize, revtime, damping, inputbw, spread, drylevel.dbamp, earlylevel.dbamp, taillevel.dbamp);
     pan    = Pan2.ar(Mix(filt +comb), pos, 1);
	OffsetOut.ar(out, (pan*env*Lag.kr(amp,lagAmp)))
}).add;

//smooth strings with no reverb incorporated
SynthDef(\em_smooth_strings_noVerb_octo, {arg freqRate=1, freqMul=200, freq=300,  lowCutfreq=22000, highCutfreq=10, gate=1, rq=1, decay=1, combDecay= 0.15, attack=0.01,amp=1.0,lagAmp=0, rel=0.3, pos=0, curve= -3, out=0, harmAmp=1, combAmp=0.5, roomSize=3, revtime=2.6, damping=0.2, inputbw=0.48, spread=15, drylevel= -3, earlylevel= -4, taillevel= -13, rqInit=0.005, rqEnd=0.0005, rqTime=0.1, freqFactor=0.02,orientation= 0.5;
	 var  noise, demand, filt, env, freqLo, freqHi, pan, comb, verb,rqEnv, freqLFO;
	 freqLFO=LFDNoise3.kr(freqRate,freqMul,freq);
	freqLo= (freqLFO) - (freqLFO*freqFactor);
	freqHi= (freqLFO) + (freqLFO*freqFactor);
	 noise = PinkNoise.ar(2.0);
	 rqEnv = EnvGen.ar(Env([rqInit, rqEnd],[rqTime]), doneAction:0);
	 env = EnvGen.ar(Env.asr(attack, 1, decay,curve ), gate, doneAction:2);
	 demand = Duty.ar(0.01, 0,Dbrown(freqLo, freqHi,1));
	 filt   = (1..30).collect{|harmonic|BPF.ar(noise, demand*harmonic, rqEnv*rq, (1/(harmonic**harmAmp)))};
     filt   = Mix(LPF.ar(filt, lowCutfreq));
     filt   = HPF.ar(filt,highCutfreq);
     comb   = CombN.ar(filt, 0.2, freqLFO.reciprocal, combDecay, combAmp );
    // verb   =  GVerb.ar(filt +comb, roomSize, revtime, damping, inputbw, spread, drylevel.dbamp, earlylevel.dbamp, taillevel.dbamp);
     //pan    = Pan2.ar(Mix(filt +comb), pos, 1);
	pan= PanAz.ar(8,Mix(filt +comb),pos,1,2,orientation);
	OffsetOut.ar(out, (pan*env*Lag.kr(amp,lagAmp)))
}).add;


SynthDef(\RO_ZitaVerb_1_mono, {|in1=30, in2=31, drywet=0.5,lowfreq=600,midfreq=6000, predelay=0.1, rt60low=3,rt60mid=3 ,releaseVerb=0, attackVerb=0, gate=1,out=0|
	var reverb,input,signal1,signal2, env;
	env = EnvGen.ar(Env.asr(attackVerb, 1,releaseVerb),gate:gate, doneAction:2);
	signal1=In.ar(in1,2);
	//signal2=In.ar(in2,1);
	reverb=Mix(ZitaRev1ST.ar(signal1[0],signal1[1],drywet,lowfreq,midfreq, predelay, rt60low,rt60mid));


	Out.ar(out,reverb);
}).add;




SynthDef(\itin_Verb3mono, {| in=30, revtime=0.50, roomSize=50, damping=0.62,inputbw=0.48,earlylevel= -11, taillevel= -13, out=0, drylevel= -3.0, spread=30,releaseVerb=0, attackVerb=0, gate=1|
	var reverb,input,signal, env;
	env = EnvGen.ar(Env.asr(attackVerb, 1,releaseVerb),gate:gate, doneAction:2);
	signal=In.ar(in,1);
	reverb=GVerb.ar(signal,roomSize, revtime, damping, inputbw, spread, drylevel.dbamp, earlylevel.dbamp, taillevel.dbamp);
	Out.ar(out,Mix(reverb));
}).add;
