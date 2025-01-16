# -*- coding: utf-8 -*-
import os

from sst_unittest import *
from sst_unittest_support import *

############################################################################


class testcase_drra(SSTTestCase):

    def setUp(self):
        super(type(self), self).setUp()
        # Put any code here that needs to be run before each test is executed

    def tearDown(self):
        # Put any code here that needs to be run after each test is executed
        super(type(self), self).tearDown()

    #####
    def test_drra_simpleExample(self):
        self.drra_template("simpleExample")

    def drra_template(self, testcase, striptotail=0, checkpoint=False):
        # Get the path to the test files
        test_path = self.get_testsuite_dir()
        outdir = self.get_test_output_run_dir()
        tmpdir = self.get_test_output_tmp_dir()

        # Set the various file paths
        testDataFileName = f"{testcase}"

        sdlfile = f"{test_path}/{testDataFileName}.py"
        reffile = f"{test_path}/refFiles/{testDataFileName}.out"
        outfile = f"{outdir}/{testDataFileName}.out"
        tmpfile = f"{tmpdir}/{testDataFileName}.tmp"
        cmpfile = f"{tmpdir}/{testDataFileName}.cmp"
        errfile = f"{outdir}/{testDataFileName}.err"
        mpioutfiles = f"{outdir}/{testDataFileName}.testfile"

        if not checkpoint:
            self.run_sst(sdlfile, outfile, errfile, mpi_out_files=mpioutfiles)

            testing_remove_component_warning_from_file(outfile)

            # Copy the output file to the cmp file
            os.system(f"cp {outfile} {cmpfile}")

            if striptotail:
                # Post processing of the output data to scrub it into a format to compare
                os.system(f"grep Random {outfile} > {tmpfile}")
                os.system(f"tail -5 {tmpfile} > {outfile}")

            # NOTE: THE PASS / FAIL EVALUATIONS ARE PORTED FROM THE SQE BAMBOO
            #       BASED testSuite_XXX.sh THESE SHOULD BE RE-EVALUATED BY THE
            #       DEVELOPER AGAINST THE LATEST VERSION OF SST TO SEE IF THE
            #       TESTS & RESULT FILES ARE STILL VALID

            # Perform the test
            if os_test_file(errfile, "-s"):
                log_testing_note(
                    f"drra test {testDataFileName} has a Non-empty Error File {errfile}"
                )

            cmp_result = testing_compare_sorted_diff(testcase, cmpfile, reffile)
            if not cmp_result:
                diffdata = testing_get_diff_data(testcase)
                log_failure(diffdata)
            self.assertTrue(
                cmp_result,
                f"Sorted Output file {cmpfile} does not match Reference file {reffile}",
            )

        # Checkpoint test
        else:
            cptfreq = "15us"
            cptrestart = "0_15000000"

            # Generate the checkpoint file
            sdlfile_generate = f"{test_path}/{testcase}.py"
            outfile_generate = f"{outdir}/{testcase}_generate.out"
            options_checkpoint = (
                f"--checkpoint-sim-period={cptfreq} --checkpoint-prefix={testcase}"
            )
            self.run_sst(
                sdlfile_generate, outfile_generate, other_args=options_checkpoint
            )

            # Run from restart
            sdlfile_restart = f"{outdir}/{testcase}/{testcase}_{cptrestart}/{testcase}_{cptrestart}.sstcpt"
            outfile_restart = f"{outdir}/{testcase}_restart.out"
            options_restart = "--load-checkpoint"
            self.run_sst(sdlfile_restart, outfile_restart, other_args=options_restart)

            # Check that restart output is a subset of checkpoint output
            cmp_result = testing_compare_filtered_subset(
                outfile_restart, outfile_generate
            )
            self.assertTrue(
                cmp_result,
                f"Output/Compare file {outfile_restart} does not match Reference file {outfile_generate}",
            )
