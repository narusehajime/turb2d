from debris_flow import DebrisFlow
from landlab.io.esri_ascii import read_esri_ascii
import numpy as np
import copy
import ipdb
import time
import multiprocessing as mp


class RunMultiFlows():
    """A class to run multiple flows for conducting inverse analysis
    """

    def __init__(
            self,
            proc,
            range_Cf,
            range_bfa,
            range_thick,
            init_loc,
            flow_volume,
            topofile_smoothed_topo_without_lobe,
            topofile_raw_topo,
            topofile_lobe_edge_masked_topo,
            topofile_raw_topo_without_lobe,
            result_obj_filename='obj.txt',
            result_init_filename='init.txt',
            endtime=100,
    ):

        # read topographic file
        self.grid, self.smoothed_topo_without_lobe = read_esri_ascii(
            topofile_smoothed_topo_without_lobe)
        # raw data whose edge is expressed as -9999
        self.raw_topo = read_esri_ascii(topofile_raw_topo)[1]
        # raw data whose edge and lobe is expressed as -9999
        self.lobe_edge_masked_topo = read_esri_ascii(
            topofile_lobe_edge_masked_topo)[1]

        # raw data but exception:slope on which lobe is estimated one
        self.raw_topo_without_lobe = read_esri_ascii(
            topofile_raw_topo_without_lobe)[1]

        # get location of lobe
        mask_edge = np.where(self.raw_topo == -9999)[0]
        mask_edgeetlobe = np.where(self.lobe_edge_masked_topo == -9999)[0]
        mask_lobe = list(set(mask_edge) ^ set(mask_edgeetlobe))
        self.mask_lobe = np.array(mask_lobe)

        # Calculate thickness of lobe
        self.lobe_thick = self.raw_topo[
            self.mask_lobe] - self.raw_topo_without_lobe[self.mask_lobe]

        # set initial parameters
        self.topofile = topofile_smoothed_topo_without_lobe
        self.range_Cf = range_Cf
        self.range_bfa = range_bfa
        self.range_thick = range_thick
        self.init_loc = init_loc
        self.flow_volume = flow_volume
        self.processors = proc
        self.result_obj_filename = result_obj_filename
        self.result_init_filename = result_init_filename
        self.endtime = endtime

    def produce_dflow(self, Cf, bfa, thick):
        """ producing a DebrisFlow object.
        """

        # read topography
        # grid = read_esri_ascii(self.topofile)[0]
        grid = copy.deepcopy(self.grid)

        # First, a grid object is created
        # grid = copy.deepcopy(self.grid)
        grid.add_field('node', 'topographic__elevation',
                       self.smoothed_topo_without_lobe)
        grid.add_zeros('flow__depth', at='node')
        grid.add_zeros('flow__horizontal_velocity', at='link')
        grid.add_zeros('flow__vertical_velocity', at='link')

        # set initial flow region
        flow_region_width = np.sqrt(self.flow_volume / thick)
        init_loc = self.init_loc + np.array([grid.node_x[0], grid.node_y[0]])
        initial_flow_region = (grid.node_x >
                               (init_loc[0] - flow_region_width / 2)) & (
                                   grid.node_x <
                                   (init_loc[0] + flow_region_width / 2)
                               ) & (grid.node_y >
                                    (init_loc[1] - flow_region_width / 2)) & (
                                        grid.node_y <
                                        (init_loc[1] + flow_region_width / 2))
        grid.at_node['flow__depth'][initial_flow_region] = thick
        grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

        # create debris flow
        dflow = DebrisFlow(
            grid,
            Cf=Cf,
            h_init=0.001,
            alpha=0.01,
            flow_type='Voellmy',
            basal_friction_angle=bfa)

        return dflow

    def get_objective_function(self, dflow):
        """Calculate objective function
        """

        computed_lobe = dflow.grid.at_node['flow__depth'][self.mask_lobe]
        obj = np.sum(self.lobe_thick - computed_lobe)**2

        return obj

    def run_flow(self, init_values):
        """ Run a flow to obtain the objective function
        """

        # Produce debris_flow object
        dflow = self.produce_dflow(init_values[0], init_values[1],
                                   init_values[2])

        # Run the model
        dt = 1.0
        num_of_loops = np.round(self.endtime / dt).astype(np.int64)
        for i in range(num_of_loops):
            dflow.run_one_step(dt=dt)
            u = dflow.grid.at_link['flow__horizontal_velocity']
            v = dflow.grid.at_link['flow__vertical_velocity']
            v2 = u**2 + v**2
            if (np.max(v2) < 0.1):
                print('flow stopped ({0:.2f}) at {1:.3f}, {2:.3f} and {3:.3f}'.
                      format(
                          np.max(v2), init_values[0], init_values[1],
                          init_values[2]))
                break
            elif (np.max(v2) > 1000):
                print('too rapid ({0:.2f}) at {1:.3f}, {2:.3f} and {3:.3f}'.
                      format(
                          np.max(v2), init_values[0], init_values[1],
                          init_values[2]))
                break

        dflow.plot_result('{0:.3f}-{1:.3f}-{2:.3f}.png'.format(
            init_values[0], init_values[1], init_values[2]))

        obj = self.get_objective_function(dflow)

        np.save(
            'obj-{0:.3f}-{1:.3f}-{2:.3f}.npy'.format(
                init_values[0], init_values[1], init_values[2]), np.array(obj))

        # with open(self.result_init_filename, 'a') as init_file:
        #     np.savetxt(init_file, np.array([init_values]))

        # with open(self.result_obj_filename, 'a') as obj_file:
        #    np.savetxt(obj_file, np.array([obj]))

        return obj

    def run_multiple_flows(self):
        """run multiple flows
        """

        Cf = self.range_Cf
        bfa = self.range_bfa
        thick = self.range_thick

        # Create list of initial values
        init_value_list = list()
        for i in range(len(Cf)):
            for j in range(len(bfa)):
                for k in range(len(thick)):
                    init_value_list.append([Cf[i], bfa[j], thick[k]])

        # run flows using multiple processors
        pool = mp.Pool(self.processors)
        obj = pool.map(self.run_flow, init_value_list)
        # obj = list(map(self.run_flow, init_value_list))

        return init_value_list, obj


if __name__ == "__main__":

    proc = 6  # number of processors to be used
    C = np.linspace(5, 65, 3)
    Cf = 3.71 / C**2
    bfa = np.linspace(0.025, 0.50, 2)
    # thick = np.linspace(2.0, 4.0, 2)
    thick = [3.0]

    rmf = RunMultiFlows(
        proc,
        Cf,
        bfa,
        thick,
        [70., 510.],
        200.,
        '../mars_topo/gg15withoutlobe1m1g1.asc',
        '../mars_topo/goodgully15.asc',
        '../mars_topo/goodgully15holeislobe1.asc',
        '../mars_topo/gg15withoutlobe1.asc',
        endtime=2.0,
    )
    start = time.time()
    init_values, obj = rmf.run_multiple_flows()
    print("elapsed time: {} sec.".format(time.time() - start))

    obj = np.array(obj)
    init_values = np.array(init_values)
    np.save('obj.npy', obj)
    np.save('init_values.npy', init_values)
