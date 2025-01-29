/**
 * @file acoustic.cpp
 * @brief The code for the acoustic binary used in the speechmatching Python
 *        package for matching spoken audio together. This code loads the
 *        pre-trained neural network, takes a WAV file with a single audio
 *        stream and 16kHz, and runs this through the network, outputting lines
 *        with each line representing a step in time, and each line containing
 *        probabilities for characters.
 */

#include <stdlib.h>

#include <gflags/gflags.h>

#include "flashlight/fl/flashlight.h"
#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/data/FeatureTransforms.h"
#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/app/asr/data/Utils.h"
#include "flashlight/app/asr/decoder/DecodeUtils.h"
#include "flashlight/app/asr/decoder/Defines.h"
#include "flashlight/app/asr/decoder/TranscriptionUtils.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/common/SequentialBuilder.h"
#include "flashlight/ext/common/Serializer.h"

/**
 * @brief Read configuration coming with the loaded model from `tempConfig`,
 *        and stores extracted and processed configuration in `networkConfig`.
 *
 * The function reads a single string containing Gflags-style definitions, for
 * example `--someFlag=value`, from `tempConfig`. It then splits each line on
 * "=", and each string coming from that split on "--", to extract the
 * key/value pairs. These extracted pairs are then stored in `networkConfig`.
 *
 * @param[in] tempConfig A map containing configuration information loaded from
 *                       the pre-trained neural network.
 * @param[out] networkConfig A map that to which the key-value data from
                             `tempConfig` is written.
 *
 * @return Returns 0.
 */
int readConfig(
    std::unordered_map<std::string, std::string>& tempConfig,
    std::unordered_map<std::string, std::string>& networkConfig
) {
    for (std::string line : fl::lib::split("\n", tempConfig[fl::app::asr::kGflags])) {
        if (line.length() == 0) {
            continue;
        }
        std::vector<std::string> res = fl::lib::split("=", line);
        if (res.size() >= 2) {
            std::string key = fl::lib::split("--", res[0])[1];
            networkConfig[key] = res[1];
        }
    }
    return 0;
}

/**
 * @brief The main function that processes the audio.
 *
 * This function parses the command-line arguments to determine what input is
 * to be expected containing audio files to be processed. It loads various
 * files required for processing. Featurs are extracted from the audio files,
 * and taken through the pre-trained model, after which probabilities for
 * characters are calculated and written to standard output or a target file.
 * This function:
 *
 * Usage (simplified):
 *  ```
 *  ./acoustic --input-filepath <wav-file> \
 *             --acoustic-model-filepath <model-bin> \
 *             --tokens-filepath <tokens.txt> \
 *             --output-filepath <out.txt>
 *
 *  # or using short flags
 *  ./acoustic -i <wav-file> -am <model-bin> -t <tokens.txt> -o <out.txt>
 *  ```
 *
 * @param[in] argc The number of command-line arguments.
 * @param[in] argv The array of command-line arguments.
 * @return Returns 0 on success, or else a non-zero value.
 */
int main(int argc, char** argv) {
    std::string inputFilepath;
    std::string outputFilepath;
    std::string acousticModelFilepath;
    std::string tokensFilepath;
    bool useStdInput(false);

    // Parse the command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--standard-input") == 0 || strcmp(argv[i], "-stdin") == 0) {
            useStdInput = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "arguments:\n";
            std::cout << "    --input-filepath/-i PATH: the .wav file to process, single channel, 16000 hertz\n";
            std::cout << "    --acoustic-model-filepath/-am PATH: the acoustic model to use, see\n";
            std::cout << "        https://github.com/flashlight/wav2letter/tree/main/recipes/rasr\n";
            std::cout << "        70M parameters https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_70Mparams.bin\n";
            std::cout << "        300M parameters https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/am_transformer_ctc_stride3_letters_300Mparams.bin\n";
            std::cout << "    --output-filepath/-o PATH: the file to write the probabilities to\n";
            std::cout << "    --tokens-filepath/-t PATH: the path to the token file, like\n";
            std::cout << "        https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/token.txt\n";
            std::cout << "    --standard-input/-stdin: read input filenames from standard input, overwrites output file.\n";
            return 0;
        }
        if (i + 1 < argc) {
            if (strcmp(argv[i], "--input-filepath") == 0 || strcmp(argv[i], "-i") == 0) {
                inputFilepath = argv[++i];
            } else if (strcmp(argv[i], "--acoustic-model-filepath") == 0 || strcmp(argv[i], "-am") == 0) {
                acousticModelFilepath = argv[++i];
            } else if (strcmp(argv[i], "--output-filepath") == 0 || strcmp(argv[i], "-o") == 0) {
                outputFilepath = argv[++i];
            } else if (strcmp(argv[i], "--tokens-filepath") == 0 || strcmp(argv[i], "-t") == 0) {
                tokensFilepath = argv[++i];
            }
        }
    }

    // Check if there are any problems with the (combinations of) given command
    // line arguments.
    bool error_with_help(false);

    if (outputFilepath.empty()) {
        std::cout << "No output filepath is given.\n";
        error_with_help = true;
    }

    if (!error_with_help && acousticModelFilepath.empty()) {
        std::cout << "No acoustic model filepath is given.\n";
        error_with_help = true;
    }

    if (!error_with_help && tokensFilepath.empty()) {
        std::cout << "No tokens filepath is given.\n";
        error_with_help = true;
    }

    if (!error_with_help && inputFilepath.empty() && !useStdInput) {
        std::cout << "Nothing to be processed.\n";
        error_with_help = true;
    }

    if (error_with_help) {
        std::cout << "See --help/-h for more information.\n";
        return 1;
    }

    // Load the pre-trained network.
    std::shared_ptr<fl::Module> network;
    std::unordered_map<std::string, std::string> tempConfig;
    std::unordered_map<std::string, std::string> networkConfig;
    std::string version;
    fl::ext::Serializer::load(acousticModelFilepath, version, tempConfig, network);
    readConfig(tempConfig, networkConfig);

    std::cout << "Loaded network for version " << version << ".\n";

    network->eval();

    // Load the tokens file.
    fl::lib::text::Dictionary dictionary(tokensFilepath);
    dictionary.addEntry(fl::app::asr::kBlankToken);

    // Prepare for extracting features from the audio file.
    fl::lib::audio::FeatureParams featureParams(
        std::atoll(networkConfig["samplerate"].c_str()),
        std::atoll(networkConfig["framesizems"].c_str()),
        std::atoll(networkConfig["framestridems"].c_str()),
        std::atoll(networkConfig["filterbanks"].c_str()),
        std::atoll(networkConfig["lowfreqfilterbank"].c_str()),
        std::atoll(networkConfig["highfreqfilterbank"].c_str()),
        std::atoll(networkConfig["mfcccoeffs"].c_str()),
        fl::app::asr::kLifterParam,
        std::atoll(networkConfig["devwin"].c_str()),
        std::atoll(networkConfig["devwin"].c_str())
    );
    featureParams.usePower = false;
    featureParams.useEnergy = false;
    featureParams.zeroMeanFrame = false;
    fl::app::asr::FeatureType featureType = fl::app::asr::FeatureType::MFSC;

    fl::Dataset::DataTransformFunction getFeatures = fl::app::asr::inputFeatures(
        featureParams,
        featureType,
        {
            networkConfig["localnrmlleftctx"] == "true",
            networkConfig["localnrmlrightctx"] == "true"
        },
        {}
    );

    while (true) {
        // Check if valid input was given.
        if (!(inputFilepath.empty() || inputFilepath.length() == 0)) {
            std::cout << "Processing file " << inputFilepath << ".\n";

            // Load the audio file.
            fl::app::asr::SoundInfo soundInfo = fl::app::asr::loadSoundInfo(
                inputFilepath.c_str());
            std::vector<float> sound = fl::app::asr::loadSound<float>(
                inputFilepath.c_str());

            // Convert the audio file to a representation of features.
            af::array features = getFeatures(
                static_cast<void*>(sound.data()),
                af::dim4(soundInfo.channels, soundInfo.frames),
                af::dtype::f32
            );

            // Take the features through the network.
            fl::Variable output = fl::ext::forwardSequentialModuleWithPadMask(
                fl::input(features),
                network,
                af::constant(features.dims(0), af::dim4(1))
            );

            // Get the probabilities using the softmax function over the output
            // from the network.
            std::vector<float> probabilities = fl::ext::afToVector<float>(
                fl::softmax(output, 0).array());

            std::cout << "Writing file " << outputFilepath << ".\n";

            int numFrames = output.dims(0);
            int timeSteps = output.dims(1);
            std::ofstream outFile(outputFilepath);

            // Write the possible characters on the first line.
            for (int i = 0; i < dictionary.indexSize(); i++) {
                outFile << dictionary.getEntry(i) << " ";
            }
            outFile << "\n";
            // Write the probabilities for each time step on each next line.
            for (int i = 0; i < timeSteps; i++) {
                for (int j = 0; j < numFrames; j++) {
                    outFile << probabilities[i*numFrames+j] << " ";
                }
                outFile << "\n";
            }
            outFile << std::endl;
            outFile.close();

            std::cout << "Processed file " << inputFilepath << ".\n";
        }

        // If not data is to be given over standard input, stop at processing
        // only the given audio file on the command line.
        if (!useStdInput) {
            break;
        }

        // ... else, wait for a new filepath to be given.
        std::cout << "Waiting for WAV file path.\n";
        std::getline(std::cin, inputFilepath);
    }

    return 0;
}

