/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;

    normal_distribution<double> dist_x(0, std[0]);
    normal_distribution<double> dist_y(0, std[1]);
    normal_distribution<double> dist_theta(0, std[2]);


    for(int i = 0; i < num_particles; i++){
        Particle p;

        p.id = i;
        p.x = x + dist_x(gen);
        p.y = y + dist_y(gen);
        p.theta = theta + dist_theta(gen);
        p.weight = 1;

        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for(int i = 0; i < num_particles; i++){
        double theta = particles[i].theta;

        if(fabs(yaw_rate) < 0.0001) {
            particles[i].x += cos(theta) * velocity;
            particles[i].y += sin(theta) * velocity;
        }
        else{
            double dt = yaw_rate*delta_t;
            particles[i].x += (velocity / yaw_rate) * (sin(theta + dt) - sin(theta));
            particles[i].y += (velocity / yaw_rate) * (cos(theta) - cos(theta + dt));
            particles[i].theta += yaw_rate * delta_t;
        }

        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].y += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < observations.size(); i++) {

        LandmarkObs observation = observations[i];

        double min_dist = numeric_limits<double>::max();
        int min_id = -1;

        for (auto predict : predicted) {

            double distance = dist(observation.x, observation.y, predict.x, predict.y);

            if (distance < min_dist) {
                min_dist = distance;
                min_id = predict.id;
            }
        }

        observations[i].id = min_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


    for (int i = 0; i < num_particles; i++) {

        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;

        vector<LandmarkObs> predicted;
        for (auto landmark : map_landmarks.landmark_list) {
            if (fabs(landmark.x_f - x) <= sensor_range && fabs(landmark.y_f - y) <= sensor_range) {
                predicted.push_back(LandmarkObs{ landmark.id_i, landmark.x_f, landmark.y_f });
            }
        }

        vector<LandmarkObs> observations_transformed;
        for (auto observation : observations) {
            double t_x = cos(theta)*observation.x - sin(theta)*observation.y + x;
            double t_y = sin(theta)*observation.x + cos(theta)*observation.y + y;
            auto ob = LandmarkObs{ observation.id, t_x, t_y };
            observations_transformed.push_back(ob);
        }

        dataAssociation(predicted, observations_transformed);

        particles[i].weight = 1.0;

        for (auto observation : observations_transformed) {

            double p_x, p_y;
            for (unsigned int k = 0; k < predicted.size(); k++) {
                if (predicted[k].id == observation.id) {
                    p_x = predicted[k].x;
                    p_y = predicted[k].y;
                    break;
                }
            }

            double sig_x = std_landmark[0];
            double sig_y = std_landmark[1];

            double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));
            double exponent= pow(p_x - observation.x, 2)/(2 * pow(sig_x,2)) + pow(p_y - observation.y,2) / (2 * pow(sig_y,2));

            double weight = gauss_norm * exp(-exponent);

            particles[i].weight *= weight;
        }
    }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<Particle> p_new;

    vector<double> weights;
    for (auto particle : particles) {
        weights.push_back(particle.weight);
    }

    uniform_int_distribution<int> dist1(0, num_particles-1);

    auto index = dist1(gen);
    double beta = 0.0;
    double mw =* max_element(weights.begin(), weights.end());
    uniform_real_distribution<double> dist2(0.0, mw);

    for (int i = 0; i < num_particles; i++) {
        beta += dist2(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        p_new.push_back(particles[index]);
    }

    particles = p_new;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
