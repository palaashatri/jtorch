package org.jtorch.demo.tensorlab;

import java.nio.file.Path;
import java.util.Map;
import org.jtorch.core.Tensor;
import org.jtorch.demo.common.SmokeTest;
import org.jtorch.serialization.Checkpoint;

/** TensorLab entry point. The same training path feeds the UI and the headless acceptance smoke test. */
public final class TensorLabApp {
    private TensorLabApp() {}

    public static void main(String[] args) {
        boolean smoke = SmokeTest.hasFlag(args, "--smoke-test") || SmokeTest.hasFlag(args, "--headless");
        Path checkpoint = Path.of(SmokeTest.option(args, "--checkpoint", "tensorlab-checkpoint.jtch"));
        long seed = Long.parseLong(SmokeTest.option(args, "--seed", "20261031"));
        int samples = Integer.parseInt(SmokeTest.option(args, "--samples", "1200"));
        if (smoke) {
            runSmoke(seed, samples, checkpoint);
            return;
        }
        TensorLabTraining training = new TensorLabTraining(seed, samples, 0.1);
        if (SmokeTest.hasFlag(args, "--resume")) {
            Checkpoint loaded = Checkpoint.load(checkpoint);
            training.loadStateDict(loaded.tensors());
            training.setEpoch(Integer.parseInt(loaded.metadata().getOrDefault("epoch", "0")));
        }
        new TensorLabWindow(training, checkpoint).show();
    }

    private static void runSmoke(long seed, int samples, Path checkpoint) {
        int epochs = 6;
        double learningRate = 0.1;

        // Deterministic initialisation: two identically seeded models must agree bit for bit.
        TensorLabTraining first = new TensorLabTraining(seed, samples, learningRate);
        TensorLabTraining second = new TensorLabTraining(seed, samples, learningRate);
        for (Map.Entry<String, Tensor> entry : first.stateDict().entrySet()) {
            Tensor other = second.stateDict().get(entry.getKey());
            SmokeTest.require(other != null, "missing parameter " + entry.getKey());
            double[] left = entry.getValue().toDoubleArray();
            double[] right = other.toDoubleArray();
            SmokeTest.require(java.util.Arrays.equals(left, right),
                    "parameter initialisation is not deterministic for " + entry.getKey());
        }

        double firstLoss = first.trainEpoch();
        double firstAccuracy = first.evaluate();
        SmokeTest.require(first.epoch() == 1, "epoch counter did not advance");
        Checkpoint.save(checkpoint, first.stateDict(),
                Map.of("app", "tensorlab", "epoch", "1", "seed", Long.toString(seed)));

        // Uninterrupted continuation.
        double secondLoss = first.trainEpoch();
        double secondAccuracy = first.evaluate();

        // Interrupted: load the epoch-1 checkpoint into a fresh process-equivalent state and continue.
        Checkpoint loaded = Checkpoint.load(checkpoint);
        SmokeTest.require(loaded.tensors().keySet().equals(first.stateDict().keySet()),
                "checkpoint keys do not match the module state dict");
        SmokeTest.require(loaded.metadata().getOrDefault("epoch", "").equals("1"),
                "checkpoint metadata did not record the epoch");
        TensorLabTraining resumed = new TensorLabTraining(seed, samples, learningRate);
        resumed.loadStateDict(loaded.tensors());
        resumed.setEpoch(1);
        double resumedLoss = resumed.trainEpoch();
        double resumedAccuracy = resumed.evaluate();
        SmokeTest.require(resumedLoss == secondLoss,
                "resume changed the epoch-2 loss: uninterrupted=" + secondLoss + " resumed=" + resumedLoss);
        SmokeTest.require(resumedAccuracy == secondAccuracy, "resume changed epoch-2 accuracy");

        double loss = resumedLoss;
        double accuracy = resumedAccuracy;
        for (int epoch = 3; epoch <= epochs; epoch++) {
            loss = resumed.trainEpoch();
            accuracy = resumed.evaluate();
        }
        SmokeTest.require(accuracy >= 0.90,
                "TensorLab accuracy " + accuracy + " is below the documented 0.90 threshold");
        SmokeTest.require(Double.isFinite(loss) && loss > 0, "final loss is invalid: " + loss);
        System.out.printf(
                "tensorlab epoch1_loss=%.6f epoch1_accuracy=%.4f epoch2_loss=%.6f resumed_delta=%.3e final_accuracy=%.4f%n",
                firstLoss, firstAccuracy, secondLoss, Math.abs(resumedLoss - secondLoss), accuracy);
        SmokeTest.pass("tensorlab", String.format(
                "epochs=%d accuracy=%.4f checkpoint=%s", resumed.epoch(), accuracy,
                checkpoint.toAbsolutePath().getFileName()));
    }
}
