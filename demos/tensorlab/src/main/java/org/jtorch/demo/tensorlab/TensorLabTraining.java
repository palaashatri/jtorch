package org.jtorch.demo.tensorlab;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.jtorch.api.Torch;
import org.jtorch.core.Tensor;
import org.jtorch.data.DataLoader;
import org.jtorch.data.Sample;
import org.jtorch.data.SyntheticDigits;
import org.jtorch.data.TensorDataset;
import org.jtorch.nn.CrossEntropyLoss;
import org.jtorch.nn.Linear;
import org.jtorch.nn.Module;
import org.jtorch.nn.ReLU;
import org.jtorch.nn.Sequential;
import org.jtorch.nn.StateDict;
import org.jtorch.optim.Optimizer;
import org.jtorch.optim.SGD;

/**
 * The single training implementation used by both TensorLab's window and its headless smoke test. No canned outputs:
 * every reported metric is computed by running the shared JTorch runtime.
 */
public final class TensorLabTraining {
    public static final int FEATURES = SyntheticDigits.FEATURES;
    public static final int CLASSES = SyntheticDigits.CLASSES;

    private final Module model;
    private final Optimizer optimizer;
    private final CrossEntropyLoss criterion = new CrossEntropyLoss();
    private final TensorDataset trainingData;
    private final TensorDataset testData;
    private final DataLoader trainingLoader;
    private final DataLoader testLoader;
    private final List<Double> lossHistory = new ArrayList<>();
    private final List<Double> accuracyHistory = new ArrayList<>();
    private int epoch;

    public TensorLabTraining(long seed, int samples, double learningRate) {
        Torch.manualSeed(seed);
        TensorDataset full = SyntheticDigits.create(samples, seed);
        int trainCount = (int) (samples * 0.8);
        Tensor images = full.columns().get(0);
        Tensor labels = full.columns().get(1);
        this.trainingData = new TensorDataset(
                images.narrow(0, 0, trainCount), labels.narrow(0, 0, trainCount));
        this.testData = new TensorDataset(
                images.narrow(0, trainCount, samples - trainCount),
                labels.narrow(0, trainCount, samples - trainCount));
        this.model = new Sequential(
                new Linear(FEATURES, 64),
                new ReLU(),
                new Linear(64, CLASSES));
        this.optimizer = SGD.builder()
                .parameters(model.parameters())
                .learningRate(learningRate)
                .build();
        this.trainingLoader = new DataLoader(trainingData, 64, true, false, seed);
        this.testLoader = new DataLoader(testData, 256, false, false, seed);
    }

    public double trainEpoch() {
        trainingLoader.setEpoch(epoch);
        double totalLoss = 0;
        long totalSamples = 0;
        for (Sample batch : trainingLoader) {
            Tensor inputs = batch.tensor(0);
            Tensor targets = batch.tensor(1);
            Tensor logits = model.call(inputs);
            Tensor loss = criterion.forward(logits, targets);
            optimizer.zeroGrad();
            loss.backward();
            optimizer.step();
            double value = loss.item();
            if (!Double.isFinite(value)) {
                throw new IllegalStateException("Training diverged at epoch " + epoch + ": loss=" + value);
            }
            totalLoss += value * inputs.size(0);
            totalSamples += inputs.size(0);
        }
        epoch++;
        double meanLoss = totalLoss / totalSamples;
        lossHistory.add(meanLoss);
        return meanLoss;
    }

    public double evaluate() {
        return Torch.noGrad(() -> {
            long correct = 0;
            long total = 0;
            for (Sample batch : testLoader) {
                Tensor inputs = batch.tensor(0);
                Tensor targets = batch.tensor(1);
                Tensor predictions = model.call(inputs).argmax(1, false);
                correct += predictions.eq(targets).sum().itemLong();
                total += inputs.size(0);
            }
            double accuracy = (double) correct / total;
            accuracyHistory.add(accuracy);
            return accuracy;
        });
    }

    public Map<String, Tensor> stateDict() {
        return model.stateDict();
    }

    public void loadStateDict(Map<String, Tensor> state) {
        model.loadStateDict(new StateDict(state), true);
    }

    public Module model() {
        return model;
    }

    public Optimizer optimizer() {
        return optimizer;
    }

    public int epoch() {
        return epoch;
    }

    public void setEpoch(int value) {
        this.epoch = value;
    }

    public List<Double> lossHistory() {
        return List.copyOf(lossHistory);
    }

    public List<Double> accuracyHistory() {
        return List.copyOf(accuracyHistory);
    }

    public TensorDataset trainingData() {
        return trainingData;
    }

    public TensorDataset testData() {
        return testData;
    }
}
