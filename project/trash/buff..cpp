   /*struct Info info;
    info.world_rank = world_rank;
    info.world_rank = world_size;
    info.iter_max = iter_max;
    info.N = N;
    info.start_T = T;
    info->iter = 0;
    info->n = 0;

    //printf("world_size = %d, world_rank = %d\n",world_size,world_rank);

    // Initialize
    double 	***u_h, ***u_d, ***uold_h;
    int iter = 0;
    int m = N + 1;
    int *Nz; // Stores dimensions for matrices

    // Get number of devices
    int num_device;
	cudaGetDeviceCount(&num_device);
    //printf("We have %d devices!\n",num_device);

    // Get local rank
    info.local_rank = world_rank % num_device;

    // Check if there are sufficient number of devices
    /*if (world_size > num_device) {
        printf("You tried to run on more GPUs than we have available!\n");
        return;
    }
    if (world_size == 1) {
        printf("It seems kinda silly to run this script for only one GPU!\n");
        return;
    } 
    //world_size = num_device;

    // Select correct device
    cudaSetDevice(local_rank);

    // Get section we need on this specific device (integer division)
    info.kstart = N/world_size*world_rank + 1;
    info.kend = world_rank == world_size - 1 ? N : N/world_size*(world_rank + 1);
    /*if (world_rank == world_size + 1) {
        kend = N+1;
    }
    //printf("(kstart, kend) = (%d,%d)\n",kstart,kend);

    // Check if we have peer access to next and previous device.
    // If it is the last device, it has the same consequence as not having peer access
    // in terms of allocation. 
    int canAccessPeerNext = false;
    int canAccessPeerPrev = false;
    if (world_rank < world_size - 1) {
        cudaDeviceCanAccessPeer (&canAccessPeerNext, world_rank, world_rank + 1);
        //if (canAccessPeerNext)
        //    printf("We have peer-access to next device! I am %d :)\n",world_rank);
    }
    if (world_rank > 0) {
        cudaDeviceCanAccessPeer (&canAccessPeerPrev, world_rank, world_rank - 1);
        //if (canAccessPeerPrev)
        //    printf("We have peer-access to previous device! I am %d :)\n",world_rank);
    }

    //printf("(%d,%d)\n",canAccessPeerNext,canAccessPeerPrev);

    int tempkstart, tempkend, tempCanAccessPeerNext, tempCanAccessPeerPrev;
    // Show information about MPI call on device 0 by sending everything to rank 0
    if (info.world_rank == 0) {
        // Print for rank 0
        printf("Rank 0:\n \t(kstart, kend) = (%d,%d)\n",info.kstart,info.kend);
        if (canAccessPeerNext) printf("\tI have peer-access to next device!\n"); // tjek af peer access skal fikses
        printf("\tI can see %d devices!\n",num_device);
        // Receive from other ranks and print
        for (int i = 1; i < world_size; i++) {
            MPI_Recv(&tempkstart,            1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&tempkend,              1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&tempCanAccessPeerPrev, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&tempCanAccessPeerNext, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Rank %d:\n \t(kstart, kend) = (%d,%d)\n",i,tempkstart,tempkend);
            if (tempCanAccessPeerPrev) printf("\tI have peer-access to previous device!\n");
            if (tempCanAccessPeerNext) printf("\tI have peer-access to next device!\n");
        }
    }
    else {
        // Send to rank 0
        MPI_Send(&info.kstart,            1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&info.kend,              1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&info.canAccessPeerPrev, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&info.canAccessPeerNext, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Width along the k-direction on a specific device
    info.kwidth = kend - kstart + 1;
    
    // Allocation on host
    u_h     = host_malloc_3d(N+2, N+2, kwidth + 2);
    uold_h  = host_malloc_3d(N+2, N+2, kwidth + 2);
    f_h     = host_malloc_3d(N+2, N+2, kwidth + 2);

    // Check allocation
    if (u_h == NULL || uold_h == NULL || f_h == NULL) {
        perror("Allocation failed on host!");
        exit(-1);
    }

    // Allocation on device
    device_malloc_3d(&info.u_d,    &info.u_log,    N+2, N+2, kwidth + 2);
    device_malloc_3d(&info.uold_d, &info.uold_log, N+2, N+2, kwidth + 2);
    device_malloc_3d(&info.f_d,    &info.f_log,    N+2, N+2, kwidth + 2);

    // Check allocation
    if (info.u_d == NULL || info.uold_d == NULL || info.f_d == NULL || info.u_log == NULL || info.uold_log == NULL || info.f_log == NULL) {
        perror("Allocation failed on device!");
        exit(-1);
    }

    // Save info about MPI and the split between GPUs
    //struct MPInfo info = {world_size, world_rank, local_rank, canAccessPeerPrev, canAccessPeerNext, kstart, kend, kwidth}; 

    // Initialize start and boundary conditions on host
    init(u_h, uold_h, f_h, N, start_T, info.kstart, info.kend);

    // Do GPU warm-up (missing!!)
    //jacobi(u_d, uold_d, f_d, N, iter_max, &tolerance, &iter);
    iter = 0;

    double start_transfer = omp_get_wtime();
    // Copy initializd array to devices
    // We move a line too much to each device such that we do not have to transfer in between before first iteration
    cudaMemcpy(info.uold_log, **uold_h, (N+2) * (N+2) * info.kwidth * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(info.u_log,    **u_h,    (N+2) * (N+2) * info.kwidth * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(info.f_log,    **f_h,    (N+2) * (N+2) * info.kwidth * sizeof(double), cudaMemcpyHostToDevice);

    // Call Jacobi iteration
    double start = omp_get_wtime();
    jacobi_mgpu(info);
    double stop = omp_get_wtime() - start;

    cudaMemcpy(**uold_h, info.uold_log, (N+2) * (N+2) * info.kwidth * sizeof(double), cudaMemcpyDeviceToHost);
    double stop_transfer = omp_get_wtime() - start_transfer;
    printf("%d %d %.5f %.5f %.5e # N iterations time transfer_time error\n", N, iter, stop, stop_transfer, info.tolerance);

    // Dump  results if wanted
    output_prefix = "poisson_mgpu"; 
    dump_output(output_type, output_prefix, output_ext, extra_str, N, uold_h);*/