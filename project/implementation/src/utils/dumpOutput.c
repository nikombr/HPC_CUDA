#include <stdio.h>
#include <stdlib.h>
#include "../../lib/print.h"

void dump_output(int output_type, char *output_prefix, char *output_ext, char *extra_str, int N, double *** u) {
    char	output_filename[FILENAME_MAX];  // Filename for output file
    switch(output_type) {
        case 0:
            // No output at all
            break;
        case 3:
            output_ext = ".bin";
            sprintf(output_filename, "../../results/%s_%d%s%s", output_prefix, N, extra_str, output_ext);
            fprintf(stderr, "\nWrite binary dump to %s\n", output_filename);
            print_binary(output_filename, N, u);
            break;
        case 4:
            output_ext = ".vtk";
            sprintf(output_filename, "../../results/%s_%d%s", output_prefix, N, output_ext);
            fprintf(stderr, "\nWrite VTK file to %s\n", output_filename);
            print_vtk(output_filename, N, u);
            break;
        default:
            fprintf(stderr, "Non-supported output type!\n");
            break;
        }
}