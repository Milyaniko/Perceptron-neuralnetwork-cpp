#include <iostream>
#include <vector>
#include "connection.cpp"
#include <cstdlib>
#include <cmath>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

class Neuron
{
  public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);

  private:
    static double transferFunction(double x);
    static double transferFanctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFanctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFanctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
    return tanh(x);
}

double transferFanctionDerivative(double x)
{
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    for(unsigned n = 0; n < prevLayer.size(); n++) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}
