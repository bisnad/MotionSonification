SynthDef(\FOF_substract_Pulse_8chan, { arg freq=200,  lpfFreq=20000,out=0,amp=1, vibRate=6, vibDepth= 0.1, lag=0.5,widthFreq=10, widthMul=0.4, widthAdd=0.5, attack= 0.1,release=0.5,gate=1,panPos=0.0, lagFreq=0.0, lagFormantFreqs=0.0, lagFormantAmps= 0.0, lagFormantQs= 0.0,lagVibDepth=0, lagAmp= 0, orientation=0.5,
        formantFreqs = #[ 400, 750, 2400, 2600, 2900 ],
        formantAmps  = #[ 1, 0.28183829312645, 0.089125093813375, 0.1, 0.01 ],
        formantQs     = #[ 0.1, 0.10666666666667, 0.041666666666667,0.046153846153846, 0.041379310344828 ];
    var vibrato, signal, filter, envelope,pan;
	vibrato=SinOsc.kr(vibRate, 0, Lag.kr(vibDepth,lagVibDepth)-1,1);
	signal = LPF.ar(Pulse.ar(Lag.kr(freq,lagFreq)*vibrato ,LFDNoise3.ar(widthFreq,widthMul,widthAdd)), lpfFreq);

filter=BBandPass.ar(signal,Lag.kr(formantFreqs,lagFormantFreqs),Lag.kr(formantQs,lagFormantQs),
		Lag.kr(formantAmps,lagFormantAmps));

	envelope=EnvGen.ar(Env.asr(attack, 1,release),gate:gate, doneAction:2);
 	pan= PanAz.ar(8,filter,panPos,1,2,orientation);
	Out.ar(out, Mix.new( pan*envelope* Lag.kr(amp, lagAmp)));
}).add;


SynthDef(\itin_Verb3mono, {| in=30, revtime=0.50, roomSize=50, damping=0.62,inputbw=0.48,earlylevel= -11, taillevel= -13, out=0, drylevel= -3.0, spread=30,releaseVerb=0, attackVerb=0, gate=1|
	var reverb,input,signal, env;
	env = EnvGen.ar(Env.asr(attackVerb, 1,releaseVerb),gate:gate, doneAction:2);
	signal=In.ar(in,1);
	reverb=GVerb.ar(signal,roomSize, revtime, damping, inputbw, spread, drylevel.dbamp, earlylevel.dbamp, taillevel.dbamp);
	Out.ar(out,Mix(reverb));
}).add;
