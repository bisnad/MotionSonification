
//Flute
SynthDef("waveguideFluteMono_lag", { arg scl = 0.2, freq = 72, ipress = 0.90, ibreath = 0.00536, ifeedbk1 = 0.4, ifeedbk2 = 0.4, dur = 1, gate = 1, amp = 0.2, attack=0.1, attackNoise=0.06, rel=0.1,relNoise=0.1, vibRate=6,vibWidth=0.1, out=0, amp2=1, lagFreq=0, lagAmp=0, lagBreath=0, lpf=2000;

	var kenv1, kenv2, kenvibr, kvibr, sr, cr, block;
	var poly, signalOut, ifqc;
	var aflow1, asum1, asum2, afqc, atemp1, ax, apoly, asum3, avalue, atemp2, aflute1;
	var fdbckArray;

	sr = SampleRate.ir;
	cr = ControlRate.ir;
	block = cr.reciprocal;

	ifqc = freq;

	// noise envelope
	kenv1 = EnvGen.kr(Env.new(
		[ 0.0, 1.1 * ipress, ipress, ipress, 0.0 ], [ attackNoise, 0.2, dur - 0.46, relNoise ], 'linear',releaseNode: 2 ), gate
	);
	// overall envelope
	kenv2 = EnvGen.kr(Env.new(
		[ 0.0, amp, amp, 0.0 ], [ attack, dur - 0.02, rel ], 'linear' ,releaseNode: 2), gate, doneAction: 2
	);
	// vibrato envelope
	kenvibr = EnvGen.kr(Env.new( [ 0.0, 0.0, 1, 1, 0.0 ], [ 0.5, 0.5, dur - 1.5, 0.5 ], 'linear',releaseNode: 2), gate );

	// create air flow and vibrato
	aflow1 = LFClipNoise.ar( sr, kenv1 );
	kvibr = LFDNoise3.ar( vibRate,  vibWidth * kenvibr,0 ); //cambié vibrato por un Noise.

	asum1 = ( Lag.kr(ibreath,lagBreath) * aflow1 ) + kenv1 + kvibr;
	afqc =Lag.kr(ifqc.reciprocal,lagFreq) - ( asum1/20000 ) - ( 9/sr ) + ( ifqc/12000000 ) - block;

	fdbckArray = LocalIn.ar( 1 );

	aflute1 = fdbckArray;
	asum2 = asum1 + ( aflute1 * ifeedbk1 );

	//ax = DelayL.ar( asum2, ifqc.reciprocal * 0.5, afqc * 0.5 );
	ax = DelayC.ar( asum2, Lag.kr(ifqc.reciprocal,lagFreq) - block * 0.5, afqc * 0.5 - ( asum1/ifqc/cr ) + 0.001 );

	apoly = ax - ( ax.cubed );
	asum3 = apoly + ( aflute1 * ifeedbk2 );
	avalue = LPF.ar( asum3, lpf );

	aflute1 = DelayC.ar( avalue, Lag.kr(ifqc.reciprocal,lagFreq) - block, afqc );

	fdbckArray = [ aflute1 ];

	LocalOut.ar( fdbckArray );

	signalOut = avalue;

	OffsetOut.ar( out, Mix.ar([ signalOut * kenv2, signalOut * kenv2 ])*Lag.kr(amp2, lagAmp ));

}).store;


// REverb


//verb3 for Pmono




SynthDef(\itin_Verb3mono, {| in=30, revtime=0.50, roomSize=50, damping=0.62,inputbw=0.48,earlylevel= -11, taillevel= -13, out=0, drylevel= -3.0, spread=30,releaseVerb=0, attackVerb=0, gate=1|
	var reverb,input,signal, env;
	env = EnvGen.ar(Env.asr(attackVerb, 1,releaseVerb),gate:gate, doneAction:2);
	signal=In.ar(in,1);
	reverb=GVerb.ar(signal,roomSize, revtime, damping, inputbw, spread, drylevel.dbamp, earlylevel.dbamp, taillevel.dbamp);
	Out.ar(out,Mix(reverb));
}).add;
