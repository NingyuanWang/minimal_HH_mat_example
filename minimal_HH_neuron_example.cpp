#include "particle_method.h"
#include "set_hh_eqn.h"
#include <chrono>
#include <boost/program_options.hpp>
#define MATLAB_VISUALIZE
bool g_display_density = true;
bool g_silence = false;
void run_HH_model(const double diffusion_coeff, const double coupling_strength, const double coupling_potential, const std::string ICfilename, const double time_stepsize, const int stepsize_count, const double tau, const double lambda, const double alpha, const double linear_tolerance, const int plot_interval, std::vector<bool> projection_dimension, std::vector<bool> density_projection_dimension, const bool generate_video) {
//Setting models:
	Population_density_with_equation* a_ptr = NULL;
	int plot_stepcount;
	if (g_display_density) {
		plot_stepcount = stepsize_count / plot_interval;
	}
	Advection_diffusion_eqn* HH = set_Hodgkin_Huxley_eqn(diffusion_coeff, coupling_strength, coupling_potential);
	double plot_xlb, plot_xub, plot_ylb, plot_yub;
	a_ptr = new Population_density_with_equation(*HH,4, tau,lambda,alpha);//NOTE: 4 is the dimension of HH model. Change this to match your model.
	a_ptr->input_all_particles(ICfilename.c_str());
	if (g_display_density)
	{
        passplotcommand(global_matlab_engine, "clear");
		passplotcommand(global_matlab_engine, "subplot(2,1,1)");
		a_ptr->plot("axis([-0.2,1.1,0,1])");
		passplotcommand(global_matlab_engine, "subplot(2,1,2)");
        a_ptr->plot_density(density_projection_dimension, -0.2, 1.1, 0.0, 1.0);
	}
	const double average_V = a_ptr->average_in_index(0);
	std::string matlabarg;
	if (g_display_density)
	{
		matlabarg = "title('average voltage : " + std::to_string(average_V * 100) + "mV')";
		passplotcommand(global_matlab_engine, matlabarg.c_str());
        if (!g_silence)
        {
            std::cout << "Problem setup finished.\n";
        }
	}
//Setting timings:
	std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
	auto t_update = t0 - t0;
	double t = 0.0;
	typedef std::chrono::duration<double, std::ratio<1, 1>> seconds;
//Setting output of average voltage and coupling_strength
	MATFile* output_matptr;
	std::string mat_filename;
	mat_filename = "coupling_strength_diff_" + std::to_string(int(100000 * diffusion_coeff)) + "_coup_" + std::to_string(int(10 * coupling_strength)) + ".mat";
	output_matptr = matOpen(mat_filename.c_str(), "w");
	mxArray *avg_potential_matlabarray;
	mxArray *coupling_strength_matlabarray;
	mxArray *t_matlabarray;
	mxArray *particle_count_matlabarray;
	avg_potential_matlabarray = mxCreateDoubleMatrix(1, stepsize_count, mxREAL);
	coupling_strength_matlabarray = mxCreateDoubleMatrix(1, stepsize_count, mxREAL);
	t_matlabarray = mxCreateDoubleMatrix(1, stepsize_count, mxREAL);
	particle_count_matlabarray = mxCreateDoubleMatrix(1, stepsize_count, mxREAL);
//Setting generation of video:
	if (g_display_density && generate_video)
	{
		matlabarg = "step_count = " + std::to_string(plot_stepcount) + ";";
		passplotcommand(global_matlab_engine, matlabarg.c_str());
		passplotcommand(global_matlab_engine, "plot_itr = 1");
		passplotcommand(global_matlab_engine, "F(step_count) = struct('cdata',[],'colormap',[]);");
	}
//Loops for distribution update: 
	for (size_t i = 0; i < stepsize_count + 1; i++)
	{
		a_ptr->update_ODE_adaptive_split(time_stepsize, 1, linear_tolerance);
		t += time_stepsize;
        if (!g_silence) {
            std::cout << "time t = " << t << std::endl;
            std::cout << "coupling: " << a_ptr->coupling_at_previous_timestep() << std::endl;
            std::cout << "Particle count (before combine): " << a_ptr->size() << std::endl;
        }

		//Copy data to mat files: 
		const double coupling_strength = a_ptr->coupling_at_previous_timestep();
		const double particle_count = double(a_ptr->size());//convert to double
		const double average_V = a_ptr->average_in_index(0);
		memcpy(static_cast<double*>(mxGetPr(avg_potential_matlabarray)) + i, &average_V, sizeof(double));
		memcpy(static_cast<double*>(mxGetPr(coupling_strength_matlabarray)) + i, &coupling_strength, sizeof(double));
		memcpy(static_cast<double*>(mxGetPr(t_matlabarray)) + i, &t, sizeof(double));
		memcpy(static_cast<double*>(mxGetPr(particle_count_matlabarray)) + i, &particle_count, sizeof(double));
		if (g_display_density && (i - 1) % plot_interval == 0)
		{
			passplotcommand(global_matlab_engine, "subplot(2,1,1)");
			a_ptr->plot(projection_dimension, "axis([-0.2,1.1,0,1,0,1]);view(10,20)");
			if (generate_video)
			{
				matlabarg = "title('t = " + std::to_string(t) + " ms')";
				passplotcommand(global_matlab_engine, matlabarg.c_str());
				passplotcommand(global_matlab_engine, "xlabel('Membrane Potential (*100 mV)')");
				passplotcommand(global_matlab_engine, "ylabel('Sodium activation subunit')");
				passplotcommand(global_matlab_engine, "zlabel('Potassium subunit')");
				passplotcommand(global_matlab_engine, "subplot(2,1,2)");
				char filename[128];
				sprintf(filename, "HHTestatstep%ic%id%i.mat", i * 20 + 1, int(10 * coupling_strength), int(100000 * diffusion_coeff));
				a_ptr->output_all_particles(filename);
			}
            a_ptr->plot_density(density_projection_dimension, -0.2, 1.1, 0.0, 1.0);
			const double average_V = a_ptr->average_in_index(0);
            if (generate_video && std::count(density_projection_dimension.begin(), density_projection_dimension.end(), true) == 1 && density_projection_dimension[0] == true)
            {
                matlabarg = "title('average voltage : " + std::to_string(average_V * 100) + "mV')";
                passplotcommand(global_matlab_engine, matlabarg.c_str());
                passplotcommand(global_matlab_engine, "ylim([0 0.35])");
                passplotcommand(global_matlab_engine, "xlabel('Membrane Potential (*100 mV)')");
                passplotcommand(global_matlab_engine, "ylabel('Density')");
            }
            else if (generate_video && std::count(density_projection_dimension.begin(), density_projection_dimension.end(), true) == 2 && density_projection_dimension[0] == true && density_projection_dimension[1] == true) {
                matlabarg = "title('average voltage : " + std::to_string(average_V * 100) + "mV')";
                passplotcommand(global_matlab_engine, matlabarg.c_str());
                passplotcommand(global_matlab_engine, "xlabel('Membrane Potential (*100 mV)')");
                passplotcommand(global_matlab_engine, "ylabel('Sodium activation subunit')");
            }
            if (generate_video)
            {
				passplotcommand(global_matlab_engine, "drawnow\nF(plot_itr) = getframe(gcf);plot_itr = plot_itr + 1;");
			}
		}
		a_ptr->combine_particles();
        if (!g_silence) {
            std::cout << "Particle count (after combine): " << a_ptr->size() << std::endl;
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            std::cout << "Computation time: " << seconds(t1 - t0).count() << " seconds." << std::endl;
        }

	}
	if (g_display_density && generate_video) {
		matlabarg = "v = VideoWriter('HH_from_delta_c_" + std::to_string(int(10 * coupling_strength)) + "_d_" +
			std::to_string(int(100000 * diffusion_coeff)) + "','Motion JPEG 2000');v.FrameRate = 15;open(v);\nfor i = 1:" + std::to_string(plot_stepcount) + "\nwriteVideo(v,F(i))\nend\nclose(v);";
		passplotcommand(global_matlab_engine, matlabarg.c_str());
	}

	a_ptr->output_all_particles("HHTestatend.mat");
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    const double seconds_elapsed = seconds(t1 - t0).count();
    if (!g_silence)
    {
        std::cout << "Total Computation time: " << seconds_elapsed << " seconds." << std::endl;
    }
	matPutVariable(output_matptr, "avg_potential", avg_potential_matlabarray);
	matPutVariable(output_matptr, "coupling", coupling_strength_matlabarray);
	matPutVariable(output_matptr, "t", t_matlabarray);
	matPutVariable(output_matptr, "particle_count", particle_count_matlabarray);
	mxDestroyArray(avg_potential_matlabarray);
	mxDestroyArray(coupling_strength_matlabarray);
	mxDestroyArray(t_matlabarray);
	mxDestroyArray(particle_count_matlabarray);
    mxArray* t_computation_matlabarray;
    t_computation_matlabarray = mxCreateDoubleMatrix(1, 1, mxREAL);
    memcpy(static_cast<double*>(mxGetPr(t_computation_matlabarray)), &seconds_elapsed, sizeof(double));
    matPutVariable(output_matptr, "t_computation", t_computation_matlabarray);
    mxDestroyArray(t_computation_matlabarray);
}
int main(int argc, char **argv) {
	double diffusion_coeff;
	double coupling_strength, coupling_potential;
	double time_stepsize;
	double linear_tolerance;
	double tau;
    double lambda;
    double alpha;
	int stepsize_count;
	int plot_interval;
	bool generate_video;
	std::string projection_dimension_str;
    std::string density_projection_dimension_str;
	std::string uICfile;
	boost::program_options::options_description params("Parameters");
	params.add_options()
		("help,h", "produce help message")
		("diffusion_coeff,d", boost::program_options::value<double>(&diffusion_coeff)->default_value(1e-5), "diffusion coefficient")
		("coupling_strength,c", boost::program_options::value<double>(&coupling_strength)->default_value(0.4), "coupling strength")
        ("coupling_potential,V", boost::program_options::value<double>(&coupling_potential)->default_value(35), "mV - coupling potential")
		("tau", boost::program_options::value<double>(&tau)->default_value(0.0001), "parameter tau. affects std deviation cutoff for combine, and diffusion regularization")
        ("lambda", boost::program_options::value<double>(&lambda)->default_value(1e-6), "Tikhnov regularization factor for diffusion velocity")
        ("alpha", boost::program_options::value<double>(&alpha)->default_value(0.2), "velocity reference distance factor, small closer to center")
		("tol", boost::program_options::value<double>(&linear_tolerance)->default_value(0.05), "Tolerance for the linear approximation to deviate")
		("stepsize,s", boost::program_options::value<double>(&time_stepsize)->default_value(0.05), "ms - maximum_stepsize")
		("stepcount,N", boost::program_options::value<int>(&stepsize_count)->default_value(1000), "number of timesteps")
		("plot_interval,i", boost::program_options::value<int>(&plot_interval)->default_value(0), "timesteps between each plot (0 if no plot)")
        ("silence",boost::program_options::bool_switch(&g_silence)->default_value(false),"silent output to std::cout during computation")
		("projection_dimensions,p", boost::program_options::value<std::string>(&projection_dimension_str)->default_value("0,1,2"), "index of dimensions to project to, can have 1, 2 or 3 entries, separated by ,")
        ("density_projection_dimensions,q", boost::program_options::value<std::string>(&density_projection_dimension_str)->default_value("0,1"), "index of dimensions to project in density map, 1 or 2 entries, separated by ,")
		("video,v", boost::program_options::bool_switch(&generate_video)->default_value(true), "generates video of all plots")
		("uICfile,I", boost::program_options::value<std::string>(&uICfile)->default_value("HH_init_cond.mat"), "filename of initial concentration conditions")
	;
	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, params), vm);
	boost::program_options::notify(vm);
	if (vm.count("help")) {
		std::cout << params << "\n";
		return 0;
	}
	//Parsing projection dimensions: 	
    std::string delimiter = ",";
    const int N = 4;
    std::vector<bool> projection_dimension(N,false);
    
    int cur_dimension;
    size_t pos = 0;
    std::string current_number;
    while ((pos = projection_dimension_str.find(delimiter)) != std::string::npos) {
        current_number = projection_dimension_str.substr(0, pos);
        projection_dimension_str.erase(0, pos + delimiter.length());
        std::stringstream parse_stream(current_number);
        parse_stream >> cur_dimension;
        projection_dimension[cur_dimension] = true;
    }
    {
        std::stringstream parse_stream(projection_dimension_str);
        parse_stream >> cur_dimension;
        projection_dimension[cur_dimension] = true;
    }
    {
        std::stringstream parse_stream(projection_dimension_str);
        parse_stream >> cur_dimension;
        projection_dimension[cur_dimension] = true;
    }
    std::vector<bool> density_projection_dimension(N, false);
    while ((pos = density_projection_dimension_str.find(delimiter)) != std::string::npos) {
        current_number = density_projection_dimension_str.substr(0, pos);
        density_projection_dimension_str.erase(0, pos + delimiter.length());
        std::stringstream parse_stream(current_number);
        parse_stream >> cur_dimension;
        density_projection_dimension[cur_dimension] = true;
    }
    {
        std::stringstream parse_stream(density_projection_dimension_str);
        parse_stream >> cur_dimension;
        density_projection_dimension[cur_dimension] = true;
    }
	if (plot_interval == 0) {
		g_display_density = false;
	}
	run_HH_model(diffusion_coeff, coupling_strength, coupling_potential, uICfile, time_stepsize, stepsize_count, tau, lambda, alpha, linear_tolerance, plot_interval, projection_dimension, density_projection_dimension, generate_video);
	return 0;
}