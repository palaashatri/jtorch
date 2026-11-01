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
import org.jtorch.nn.Conv2d;
import org.jtorch.nn.CrossEntropyLoss;
import org.jtorch.nn.Flatten;
import org.jtorch.nn.Linear;
import org.jtorch.nn.MaxPool2d;
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
    private final boolean convolutional;
    private final int inputSide;
    private final List<Double> lossHistory = new ArrayList<>();
    private final List<Double> accuracyHistory = new ArrayList<>();
    private int epoch;

    public TensorLabTraining(long seed, int samples, double learningRate) {
        this(seed, samples, learningRate, false);
    }

    public TensorLabTraining(long seed, int samples, double learningRate, boolean convolutional) {
        this(splitTraining(seed, samples), splitTest(seed, samples), seed, learningRate, convolutional, 0.0);
    }

    /** Builds a training run over explicit train/test datasets (for example real MNIST IDX files). */
    public static TensorLabTraining forDatasets(
            TensorDataset training, TensorDataset test, long seed, double learningRate, boolean convolutional) {
        return new TensorLabTraining(training, test, seed, learningRate, convolutional, 0.0);
    }

    public static TensorLabTraining forDatasets(
            TensorDataset training,
            TensorDataset test,
            long seed,
            double learningRate,
            boolean convolutional,
            double momentum) {
        return new TensorLabTraining(training, test, seed, learningRate, convolutional, momentum);
    }

    private TensorLabTraining(
            TensorDataset training,
            TensorDataset test,
            long seed,
            double learningRate,
            boolean convolutional,
            double momentum) {
        Torch.manualSeed(seed);
        this.trainingData = training;
        this.testData = test;
        long features = training.columns().get(0).size(1);
        int side = (int) Math.round(Math.sqrt(features));
        if (convolutional && (long) side * side != features) {
            throw new IllegalArgumentException("CNN input must be square, got " + features + " features");
        }
        this.convolutional = convolutional;
        this.inputSide = side;
        if (convolutional) {
            int pooled = side / 2;
            this.model = new Sequential(
                    new Conv2d(1, 8, new int[] {3, 3}, new int[] {1, 1}, new int[] {1, 1}),
                    new ReLU(),
                    new MaxPool2d(2, 2, 0),
                    new Flatten(),
                    new Linear(8 * pooled * pooled, CLASSES));
        } else {
            this.model = new Sequential(
                    new Linear(FEATURES, 64),
                    new ReLU(),
                    new Linear(64, CLASSES));
        }
        this.optimizer = SGD.builder()
                .parameters(model.parameters())
                .learningRate(learningRate)
                .momentum(momentum)
                .build();
        this.trainingLoader = new DataLoader(trainingData, 64, true, false, seed);
        this.testLoader = new DataLoader(testData, 256, false, false, seed);
    }

    private static TensorDataset splitTraining(long seed, int samples) {
        TensorDataset full = SyntheticDigits.create(samples, seed);
        int trainCount = (int) (samples * 0.8);
        return new TensorDataset(
                full.columns().get(0).narrow(0, 0, trainCount),
                full.columns().get(1).narrow(0, 0, trainCount));
    }

    private static TensorDataset splitTest(long seed, int samples) {
        TensorDataset full = SyntheticDigits.create(samples, seed);
        int trainCount = (int) (samples * 0.8);
        return new TensorDataset(
                full.columns().get(0).narrow(0, trainCount, samples - trainCount),
                full.columns().get(1).narrow(0, trainCount, samples - trainCount));
    }

    private Tensor prepareInputs(Tensor inputs) {
        return convolutional ? inputs.reshape(inputs.size(0), 1, inputSide, inputSide) : inputs;
    }

    public double trainEpoch() {
        trainingLoader.setEpoch(epoch);
        double totalLoss = 0;
        long totalSamples = 0;
        for (Sample batch : trainingLoader) {
            Tensor inputs = prepareInputs(batch.tensor(0));
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
                Tensor inputs = prepareInputs(batch.tensor(0));
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
