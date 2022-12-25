# Train one model for each task
from src.behavioural_cloning_original import behavioural_cloning_train


def main():
    # print("===Training FindCave model===")
    # behavioural_cloning_train(
    #     data_dir="../basalt_neurips_data/MineRLBasaltFindCave-v0",
    #     in_model="data/VPT-models/1x.model",
    #     in_weights="data/VPT-models/foundation-model-1x.weights",
    #     out_weights="train/MineRLBasaltFindCave.weights"
    # )
    print('TRAIN ORIGINAL')
    print("===Training MakeWaterfall model===")
    behavioural_cloning_train(
        data_dir="../basalt_neurips_data/MineRLBasaltMakeWaterfall-v0",
        in_model="data/VPT-models/2x.model",
        in_weights="data/VPT-models/foundation-model-2x.weights",
        out_weights="train/foundation-model-2x-MineRLBasaltMakeWaterfall.weights"
    )
    #
    # print("===Training CreateVillageAnimalPen model===")
    # behavioural_cloning_train(
    #     data_dir="../basalt_neurips_data/MineRLBasaltCreateVillageAnimalPen-v0",
    #     in_model="data/VPT-models/foundation-1x.model",
    #     in_weights="data/VPT-models/foundation-model-1x.weights",
    #     out_weights="train/MineRLBasaltCreateVillageAnimalPen.weights"
    # )
    #
    # print("===Training BuildVillageHouse model===")
    # behavioural_cloning_train(
    #     data_dir="../basalt_neurips_data/MineRLBasaltBuildVillageHouse-v0",
    #     in_model="data/VPT-models/foundation-1x.model",
    #     in_weights="data/VPT-models/foundation-model-1x.weights",
    #     out_weights="train/MineRLBasaltBuildVillageHouse.weights"
    # )


if __name__ == "__main__":
    main()
