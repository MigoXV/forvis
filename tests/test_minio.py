from forvis.datasets.minio_driver import MinIOImageDriver,MinIOConfig

def main():
    minio_config = MinIOConfig(
        endpoint= "192.168.0.222:39000",
        access_key="test-dataset",
        secret_key="test-dataset",
        secure=False,
        bucket="forvis"
    )
    driver = MinIOImageDriver(minio_config)
    image = driver.get_image("organ/brain/brain_1-1.jpg")
    print(image.shape)
    
if __name__ == "__main__":
    main()