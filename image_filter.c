#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define TOPOLOGY_TAG 1
#define SOBEL_TAG 2
#define MEAN_TAG 3
#define TERMINATION_TAG 4

typedef struct
{
    int rank;
    int parent;
    int *children;
    int children_count;
}Process;

typedef struct
{
    int max_value;
    int width;
    int height;
    int **pixels;
}Image;

int** create_image_array(int height, int width);
void free_image_array(int **array, int height);
void split_image(Process *p, Image *image, int tag);
void gather_results(Process *p, Image *image, int tag);
void apply_filter(Image *image, int tag);

int main(int argc, char **argv)
{
    int junk, step;
    int i, j, k;
    char dump[100];
    char *parser;
    Image image;
    Process p;
    int proc_count = 0;
    int img_count = 0;
    MPI_Status status;
    int *statistic, *partial_statistic;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &(p.rank));
	MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    p.children = (int*)calloc(proc_count, sizeof(int));
    statistic = (int*)calloc(proc_count, sizeof(int));
    partial_statistic = (int*)calloc(proc_count, sizeof(int));

    FILE *topology_file = fopen(argv[1], "r");

    // Get to the line of the current process
    for (i = 0; i < p.rank; i++)
        fgets(dump, 99, topology_file);
    fgets(dump, 99, topology_file);

    // Get the neighbours of the current process
    p.children_count = 0;
    parser = strtok(dump, " \n");
    parser = strtok(NULL, " \n"); // move over the process id
    while (parser != NULL)
    {
        p.children[p.children_count++] = atoi(parser);
        parser = strtok(NULL, " \n");
    }

    // Close the file
    fclose(topology_file);

    // Determine the parent of each process in the current topology
    p.parent = -1;
    if (p.rank == 0) // Process 0 is root
    {
        junk = 1;
        for (i = 0; i < p.children_count; i++)
            MPI_Send(&junk, 1, MPI_INT, p.children[i], TOPOLOGY_TAG, MPI_COMM_WORLD);
    }
    else
    {
        // Wait for wake-up call
        MPI_Recv(&junk, 1, MPI_INT, MPI_ANY_SOURCE, TOPOLOGY_TAG, MPI_COMM_WORLD, &status);
        p.parent = status.MPI_SOURCE;

        // Wake up the children
        for (i = 0; i < p.children_count; i++)
        {
            // Remove parent from children list
            // Children must remain in order for correct statistic
            if (p.children[i] == p.parent)
            {
                for (j = i; j < p.children_count - 1; j++)
                    p.children[j] = p.children[j + 1];
                p.children_count--;
            }
            MPI_Send(&junk, 1, MPI_INT, p.children[i], TOPOLOGY_TAG, MPI_COMM_WORLD);
        }
    }

    if (p.rank == 0)
    {
        FILE *tasks_file = fopen(argv[2], "r");
        fscanf(tasks_file, "%d", &img_count);
        char filter[20];

        // READ EACH IMAGE
        for (k = 0; k < img_count; k++)
        {
            fscanf(tasks_file, "%s", filter);

            if (strcmp(filter, "mean") == 0) // go over "removal"
                fscanf(tasks_file, "%s", dump);

            char img_file_name[100];
            char result_file_name[100];
            fscanf(tasks_file, "%s", img_file_name);
            fscanf(tasks_file, "%s", result_file_name);

            // Open corresponding files
            FILE *img_file = fopen(img_file_name, "r");
            FILE *filtered_img = fopen(result_file_name, "w");

            // Header copy
            fgets(dump, 99, img_file);
            fputs(dump, filtered_img);
            fgets(dump, 99, img_file);
            fputs(dump, filtered_img);

            // Read pixels
            fscanf(img_file, "%d%d", &image.width, &image.height);
            fscanf(img_file, "%d", &image.max_value);

            // Reserve memory for the image + borders
            image.pixels = create_image_array(image.height + 2, image.width + 2);

            // Read pixels from file
            for (i = 1; i <= image.height; i++)
                for (j = 1; j <= image.width; j++)
                    fscanf(img_file, "%d", &image.pixels[i][j]);

            fclose(img_file);

            int tag;
            if (strcmp(filter, "sobel") == 0)
                tag = SOBEL_TAG;
            else
                tag = MEAN_TAG;

            if (p.children_count == 0) // if there's only one process
            {
                apply_filter(&image, tag);
                statistic[p.rank] += image.height;
            }
            else
            {
                split_image(&p, &image, tag);
                gather_results(&p, &image, tag);
            }

            // Write the results
            fprintf(filtered_img, "%d %d\n", image.width, image.height);
            fprintf(filtered_img, "%d\n", image.max_value);
            for (i = 1; i <= image.height; i++)
                for (j = 1; j <= image.width; j++)
                    fprintf(filtered_img, "%d\n", image.pixels[i][j]);

            // Free image memory
            free_image_array(image.pixels, image.height + 2);
            fclose(filtered_img);
        }

        fclose(tasks_file);

        // Signal the other processes to stop
        junk = 1;
        for (i = 0; i < p.children_count; i++)
            MPI_Send(&junk, 1, MPI_INT, p.children[i], TERMINATION_TAG, MPI_COMM_WORLD);

        // Collect statistics
        for (i = 0; i < p.children_count; i++)
        {
            MPI_Recv(partial_statistic, proc_count, MPI_INT, p.children[i],
                    TERMINATION_TAG, MPI_COMM_WORLD, &status);
            for (j = 0; j < proc_count; j++)
                if (partial_statistic[j] != 0)
                    statistic[j] = partial_statistic[j];
        }
    }
    else
    {
        while (1)
        {
            // Wait for a wake-up call to determine what action to take
            // based on the tag
            MPI_Recv(&junk, 1, MPI_INT, p.parent, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int tag = status.MPI_TAG;

            if (tag == TERMINATION_TAG)
            {
                // If the process should terminate, announce all the children
                // and collect the data from them
                junk = 1;
                for (i = 0; i < p.children_count; i++)
                    MPI_Send(&junk, 1, MPI_INT, p.children[i], TERMINATION_TAG, MPI_COMM_WORLD);
                for (i = 0; i < p.children_count; i++)
                {
                    MPI_Recv(partial_statistic, proc_count, MPI_INT, p.children[i],
                            TERMINATION_TAG, MPI_COMM_WORLD, &status);
                    for (j = 0; j < proc_count; j++)
                        if (partial_statistic[j] != 0)
                            statistic[j] = partial_statistic[j];
                }

                // Send the collected data to the parent
                MPI_Send(statistic, proc_count, MPI_INT, p.parent, TERMINATION_TAG, MPI_COMM_WORLD);
                break;
            }

            // Receive the dimensions of the chunk to process
            MPI_Recv(&image.height, 1, MPI_INT, p.parent, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(&image.width, 1, MPI_INT, p.parent, tag, MPI_COMM_WORLD, &status);

            // Reserve memory for the chunk + borders
            image.pixels = create_image_array(image.height + 2, image.width + 2);

            // Receive the image chunk line by line
            for (i = 0; i < image.height + 2; i++)
                MPI_Recv(image.pixels[i], image.width + 2, MPI_INT, p.parent,
                        tag, MPI_COMM_WORLD, &status);

            if (p.children_count == 0) // leaf nodes apply filters
            {
                apply_filter(&image, tag);
                statistic[p.rank] += image.height;
            }
            else
            {
                split_image(&p, &image, tag);
                gather_results(&p, &image, tag);
            }

            // Send the result back to the parent
            for (i = 1; i <= image.height; i++)
                MPI_Send(image.pixels[i], image.width + 2, MPI_INT, p.parent,
                        tag, MPI_COMM_WORLD);

            // Free image memory
            free_image_array(image.pixels, image.height + 2);
        }
    }

    if (p.rank == 0)
    {
        FILE *statistics_file = fopen(argv[3], "w");
        for (i = 0; i < proc_count; i++)
            fprintf(statistics_file, "%d: %d\n", i, statistic[i]);
        fclose(statistics_file);
    }

    free(p.children);
    free(partial_statistic);
    free(statistic);
	MPI_Finalize();
    return 0;
}

/**
 * Splits the image into chunks and sends them to the child processes
 * for processing.
 * @method gather_results
 * @param  p              The current process.
 * @param  image          Image chunk to process.
 * @param  tag            Communication tag to use.
 */
void split_image(Process *p, Image *image, int tag)
{
    int aux, i, j;
    int step = image->height / p->children_count;

    // When children_count > image_height
    int min = p->children_count > image->height ? image->height : p->children_count - 1;
    if (step == 0)
        step++;

    for (i = 0; i < min; i++)
    {
        // Send message to announce the filter type(tag)
        aux = 1;
        MPI_Send(&aux, 1, MPI_INT, p->children[i], tag, MPI_COMM_WORLD);

        // Send the dimnsions of the chunk
        MPI_Send(&step, 1, MPI_INT, p->children[i], tag, MPI_COMM_WORLD);
        MPI_Send(&image->width, 1, MPI_INT, p->children[i], tag, MPI_COMM_WORLD);

        // Send chunk + borders line by line
        for (j = 0; j < step + 2; j++)
            MPI_Send(image->pixels[step * i + j], image->width + 2, MPI_INT,
                    p->children[i], tag, MPI_COMM_WORLD);
    }

    if (min == image->height)
        return;

    // Last child process gets all the remaining lines
    MPI_Send(&aux, 1, MPI_INT, p->children[i], tag, MPI_COMM_WORLD);

    aux = image->height - i * step;
    MPI_Send(&aux, 1, MPI_INT, p->children[i], tag, MPI_COMM_WORLD);
    MPI_Send(&image->width, 1, MPI_INT, p->children[i], tag, MPI_COMM_WORLD);

    for (j = 0; step * i + j < image->height + 2; j++)
        MPI_Send(image->pixels[step * i + j], image->width + 2, MPI_INT,
                p->children[i], tag, MPI_COMM_WORLD);
}

/**
 * Collects the processed chunks from the child processes.
 * @method gather_results
 * @param  p              The current process.
 * @param  image          Image chunk processed.
 * @param  tag            Communication tag to use.
 */
void gather_results(Process *p, Image *image, int tag)
{
    int i, j;
    MPI_Status status;
    int step = image->height / p->children_count;

    // When children_count > image_height
    int min = p->children_count > image->height ? image->height : p->children_count - 1;
    if (step == 0)
        step++;

    for (i = 0; i < min; i++)
    {
        // Receive chunk line by line
        for (j = 1; j <= step; j++)
            MPI_Recv(image->pixels[step * i + j], image->width + 2, MPI_INT,
                    p->children[i], tag, MPI_COMM_WORLD, &status);
    }

    if (min == image->height)
        return;

    // Last child process will have all the remaining lines
    for (j = 1; step * i + j <= image->height; j++)
        MPI_Recv(image->pixels[step * i + j], image->width + 2, MPI_INT,
                p->children[i], tag, MPI_COMM_WORLD, &status);
}

/**
 * Applies the filter specified by the tag to the given image.
 * @method apply_filter
 * @param  image        The original image.
 * @param  tag          A tag associated with a filter.
 */
void apply_filter(Image *image, int tag)
{
    int i, j;
    int sobel[3][3] = { {1, 0, -1}, {2, 0, -2}, {1, 0, -1} };
    int mean[3][3] = { {-1, -1, -1}, {-1, 9, -1}, {-1, -1, -1} };
    int filter[3][3];
    int offset;

    // Choose the filter based on the incoming tag
    if (tag == SOBEL_TAG)
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                filter[i][j] = sobel[i][j];
        offset = 127;
    }
    else
    {
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                filter[i][j] = mean[i][j];
        offset = 0;
    }

    // Create a copy of the original image
    int **tmp_img = create_image_array(image->height + 2, image->width + 2);

    for (i = 0; i < image->height + 2; i++)
        for (j = 0; j < image->width + 2; j++)
            tmp_img[i][j] = image->pixels[i][j];

    // Apply the specified filter pixel by pixel
    for (i = 1; i <= image->height; i++)
        for (j = 1; j <= image->width; j++)
        {
            image->pixels[i][j] = tmp_img[i - 1][j - 1] * filter[0][0] +
                tmp_img[i - 1][j] * filter[0][1] + tmp_img[i - 1][j + 1] * filter[0][2] +
                tmp_img[i][j - 1] * filter[1][0] + tmp_img[i][j] * filter[1][1] +
                tmp_img[i][j + 1] * filter[1][2] + tmp_img[i + 1][j - 1] * filter[2][0] +
                tmp_img[i + 1][j] * filter[2][1] + tmp_img[i + 1][j + 1] * filter[2][2] + offset;

            // Bound pixel values in [0, 255]
            if (image->pixels[i][j] > 255)
                image->pixels[i][j] = 255;
            if (image->pixels[i][j] < 0)
                image->pixels[i][j] = 0;
        }

    // Free the copy
    free_image_array(tmp_img, image->height + 2);
}

/**
 * Function that creates a new 2d array of height x width size.
 */
int** create_image_array(int height, int width)
{
    int i;
    int **new_img = (int**)calloc(height, sizeof(int*));
    for (i = 0; i < height; i++)
        new_img[i] = (int*)calloc(width, sizeof(int));
    return new_img;
}

/**
 * Frees the allocated memory
 */
void free_image_array(int **array, int height)
{
    int i;
    for (i = 0; i < height; i++)
        free(array[i]);
    free(array);
}
