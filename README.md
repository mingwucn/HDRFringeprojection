# HDR+Fringe projection
## Goal

The fringe projection technique is a key method in manufacturing for acquiring 3D topography of objects. However, measuring highly reflective rotary surfaces poses challenges for optical methods, hindering the following defect detection. 
To address this, a study developed a HDR + fringe projection system, synthesizing multiple images to capture clear, detailed profiles of such surfaces for industrial products. 

## Motivation

Industrial products like aviation bearing steel balls and turbines often feature highly reflective rotary surfaces, characterized by smoothness and strong reflectivity. 

Optical measurements encounter challenges with these surfaces, leading to overexposure and underexposure areas, resulting in missing data and inaccurate measurements.
![image](https://github.com/mingwucn/HDRFringeprojection/assets/56167956/ba4bfbc7-aef0-4588-a92f-50cd377e2041)


HDR techniques can effectively address overexposure and underexposure issues in optical measurements. 
By capturing multiple images at different exposure levels and combining them, HDR extends the dynamic range of the resulting image. 

## Approach
### Intensity calibration
Variations in light intensity across the measurement area can occur due to:
- Non-uniformity of the projected light source
- Imperfections in the lens
 ![image](https://github.com/mingwucn/HDRFringeprojection/assets/56167956/767ea675-df79-4bac-b628-bed79286e3c4)

### Figure capturing with different exposure level
![image](https://github.com/mingwucn/HDRFringeprojection/assets/56167956/3aa6eb00-31a2-46cb-9409-ee8a86776a24)

### 3D profile reconstruction based on different exposure level
![image](https://github.com/mingwucn/HDRFringeprojection/assets/56167956/913588d6-58ff-4f79-8001-11f69045d0da)

## Results
![image](https://github.com/mingwucn/HDRFringeprojection/assets/56167956/76ba04d8-a2fc-4343-acc0-4be6e1ef82ab)

- HDR facilitates more accurate measurements by providing a balanced representation of the entire surface, despite its reflective properties.

- Despite its effectiveness in mitigating overexposure, HDR imaging may have limited improvement in underexposed areas, such as dark regions. This limitation could be attributed to shadows cast by geometric structures, which can offset the HDR's effectiveness in enhancing darker areas. 


