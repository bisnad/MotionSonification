ZitaRev1ST : MultiOutUGen
{
	*ar { arg in1, in2, drywet(0.5), lowfreq(600.0), midfreq(6000.0), predelay(0.1), rt60low(3.0), rt60mid(3.0), mul(1), add(0);
		^this.multiNew('audio', in1, in2, drywet, lowfreq, midfreq, predelay, rt60low, rt60mid).madd(mul, add);
  }

	*kr { arg in1, in2, drywet(0.5), lowfreq(600.0), midfreq(6000.0), predelay(0.1), rt60low(3.0), rt60mid(3.0), mul(1), add(0);
		^this.multiNew('control', in1, in2, drywet, lowfreq, midfreq, predelay, rt60low, rt60mid).madd(mul, add);
  }

  checkInputs {
    if (rate == 'audio', {
      2.do({|i|
        if (inputs.at(i).rate != 'audio', {
          ^(" la entrada en el índice " + i + "(" + inputs.at(i) +
            ") no es audio rate");
        });
      });
    });
    ^this.checkValidInputs
  }

  init { | ... theInputs |
      inputs = theInputs
      ^this.initOutputs(2, rate)
  }

  name { ^"ZitaRev1ST" }
  info { ^"Generado con Faust" }
}

