using System.Linq;
System.Action<string> print = System.Console.WriteLine;
System.Globalization.CultureInfo cu = System.Globalization.CultureInfo.InvariantCulture;

// 1. load neural network
string[] nnData = System.IO.File.ReadAllLines(@"C:\RegressionDemo\regression_8_10_10_1.txt");
// 2. create neural network from first line
int[] u = nnData[0].Split(',').Select(int.Parse).ToArray();
// 3. init neural network
float[] neuron = new float[u.Sum() + 1], weight = new float[nnData.Length - 1];
// 4. load trained weights and skip first line
for (int n = 1; n < nnData.Length; n++) weight[n - 1] = float.Parse(nnData[n]);
// 5. load test data
string[] files = System.IO.File.ReadAllLines(@"C:\RegressionDemo\houses_test.txt");
// 6. test inference - Note: optional function parm must be not zero for inference
for (int i = 0; i < files.Length; i++) NeuralRegression(files[i], 1337);
// 7. sample prediction - Note: label on position 5 must be zero
float predictionSample = NeuralRegression("-1, 0.2300, 0, 0, 1, 0, 0, 1, 0");

print("Inference House price demo\n");
print("Load Neural Network: " + string.Join("-", u));
print("Test accuracy (within 0.10) = " + (neuron[^1] * 100 / files.Length).ToString("F2"));
print("\nPredicting price for AC=no, sqft=2300," + "\n style=colonial, school=kennedy:");
print("$" + (predictionSample * 1000000).ToString("F2"));

float NeuralRegression(string inputSample, float label = 0, float net = 0)
{
    // split input string, get label, feed input data as float
    string[] token = inputSample.Split(label == 0 ? ',' : '\t');
    label = float.Parse(token[5], cu);
    for (int k = 0; k < token.Length; k++)
        neuron[(k > 5 ? k - 1 : k)] = (k != 5 ? float.Parse(token[k], cu.NumberFormat) : 0);
    // run neural regression
    for (int i = 0, t = 0, w = 0, j = u[0], L = u.Length - 1; i < L; i++, t += u[i - 1], w += u[i] * u[i - 1])
        for (int k = 0, right = u[i + 1], left = t + u[i]; k < right; k++, j++, net = 0)
        {
            for (int n = t, m = w + k; n < left; n++, m += right)
                net += neuron[n] * weight[m];
            neuron[j] = net > 0 || i == L - 1 ? net : 0;
        }
    // check result, return prediction
    neuron[^1] += System.Math.Abs(neuron[^2] - label) < 0.1f * label ? 1 : 0;
    return neuron[^2]; // return prediction (^2 == neuron.Length - 2)
}