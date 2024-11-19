import numpy as np
import cv2
def calculate_sr_hw_split_ratio(
    HB_m_offset_list, HB_n_offset_list, 
    HB_m_scale_list, HB_n_scale_list
):
    """
    Calculate SR_hw_split_ratio without overlapping regions.

    Args:
        HB_m_offset_list (List[float]): Offsets of bounding boxes in the horizontal dimension.
        HB_n_offset_list (List[float]): Offsets of bounding boxes in the vertical dimension.
        HB_m_scale_list (List[float]): Scales of bounding boxes in the horizontal dimension.
        HB_n_scale_list (List[float]): Scales of bounding boxes in the vertical dimension.

    Returns:
        str: SR_hw_split_ratio based on the condition checks.
    """
    def has_overlap(offset_list, scale_list):
        """
        Check if any boxes in the given dimension overlap.

        Args:
            offset_list (List[float]): Offsets of bounding boxes in the dimension.
            scale_list (List[float]): Scales of bounding boxes in the dimension.

        Returns:
            bool: True if there is overlap, False otherwise.
        """
        for i in range(len(offset_list)):
            for j in range(i + 1, len(offset_list)):
                if not (offset_list[i] + scale_list[i] <= offset_list[j] or
                        offset_list[j] + scale_list[j] <= offset_list[i]):
                    return True
        return False

    def redistribute_regions(offset_list, scale_list):
        """
        Redistribute the regions to ensure no overlap and full coverage.

        Args:
            offset_list (List[float]): Offsets of bounding boxes.
            scale_list (List[float]): Scales of bounding boxes.

        Returns:
            List[float]: Adjusted proportions for each region.
        """
        adjusted_ratios = []

        for i in range(len(offset_list)):
            if i == 0:
                split_ratio = offset_list[i] + scale_list[i] + (offset_list[i + 1] - offset_list[i] - scale_list[i]) / 2
                adjusted_ratios.append(split_ratio)
            elif i+1 < len(offset_list):
                mid_point = offset_list[i] + scale_list[i] + (offset_list[i + 1] - offset_list[i] - scale_list[i]) / 2
                region_ratio = mid_point - sum(adjusted_ratios)
                adjusted_ratios.append(region_ratio)
            else:
                final_ratio = 1.0 - sum(adjusted_ratios)
                adjusted_ratios.append(final_ratio)

        normalized_ratios = [ratio / sum(adjusted_ratios) for ratio in adjusted_ratios]

        return normalized_ratios

    def generate_regions(adjusted_ratios, separator):
        """
        Generate normalized regions as a string.

        Args:
            adjusted_ratios (List[float]): Adjusted proportions for each region.
            separator (str): Separator for the output string.

        Returns:
            str: Normalized regions as a string.
        """
        return separator.join(f"{region:.2f}" for region in adjusted_ratios)

    # Check for overlaps
    vertical_overlap = has_overlap(HB_m_offset_list, HB_m_scale_list)
    horizontal_overlap = has_overlap(HB_n_offset_list, HB_n_scale_list)

    # Determine which SR_hw_split_ratio to return
    if not vertical_overlap and horizontal_overlap:
        adjusted_ratios = redistribute_regions(HB_m_offset_list, HB_m_scale_list)
        return generate_regions(adjusted_ratios, ",")
    elif vertical_overlap and not horizontal_overlap:
        adjusted_ratios = redistribute_regions(HB_n_offset_list, HB_n_scale_list)
        return generate_regions(adjusted_ratios, ";")
    elif not vertical_overlap and not horizontal_overlap:
        adjusted_ratios = redistribute_regions(HB_m_offset_list, HB_m_scale_list)
        return generate_regions(adjusted_ratios, ",")
    else:
        raise ValueError("Invalid condition: Both dimensions either overlap or do not overlap.")


def generate_parameters(bbox_inputs, prompt_width, prompt_height):
    """
    Converts bbox_inputs to offset and scale lists for HB format.
    
    Args:
        bbox_inputs (List[List[int]]): List of bounding boxes, each defined by [x1, y1, x2, y2].
        prompt_width (int): Width of the entire image.
        prompt_height (int): Height of the entire image.
    
    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: 
            HB_m_offset_list, HB_n_offset_list, HB_m_scale_list, HB_n_scale_list.
    """
    HB_m_offset_list = [box[0] / prompt_width for box in bbox_inputs]
    HB_n_offset_list = [box[1] / prompt_height for box in bbox_inputs]
    HB_m_scale_list = [(box[2] - box[0]) / prompt_width for box in bbox_inputs]
    HB_n_scale_list = [(box[3] - box[1]) / prompt_height for box in bbox_inputs]

    SR_hw_split_ratio = calculate_sr_hw_split_ratio(HB_m_offset_list, HB_n_offset_list, HB_m_scale_list, HB_n_scale_list)
    

    return HB_m_offset_list, HB_n_offset_list, HB_m_scale_list, HB_n_scale_list, SR_hw_split_ratio


def visualize(HB_m_offset_list, HB_n_offset_list, HB_m_scale_list, HB_n_scale_list, SR_hw_split_ratio, prompt_width, prompt_height):
    # 创建一个白色背景的图像
    image = np.ones((prompt_height, prompt_width, 3), dtype=np.uint8) * 255

    for m_offset, n_offset, m_scale, n_scale in zip(HB_m_offset_list, HB_n_offset_list, HB_m_scale_list, HB_n_scale_list):
        x = int(m_offset * prompt_width)
        y = int(n_offset * prompt_height)
        width = int(m_scale * prompt_width)
        height = int(n_scale * prompt_height)
        # 绘制边界框
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)

    if ',' in SR_hw_split_ratio:
        split_ratios = [float(ratio) for ratio in SR_hw_split_ratio.split(',')]
        orientation = 'vertical'
    elif ';' in SR_hw_split_ratio:
        split_ratios = [float(ratio) for ratio in SR_hw_split_ratio.split(';')]
        orientation = 'horizontal'
    else:
        split_ratios = [float(SR_hw_split_ratio)]
        orientation = 'horizontal'

    colors = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (125, 125, 0), (255, 0, 255),(0, 125, 255), (125, 255, 0), (255, 255, 125), (125, 0, 0), (125, 0, 255)]
    current_pos = 0

    if orientation == 'vertical':
        total_length = prompt_width
        for i, ratio in enumerate(split_ratios):
            region_width = int(ratio * total_length)
            # 绘制分割区域
            cv2.rectangle(image, (current_pos, 0), (current_pos + region_width, prompt_height), colors[i % len(colors)], 2)
            current_pos += region_width
    else:
        total_length = prompt_height
        for i, ratio in enumerate(split_ratios):
            region_height = int(ratio * total_length)
            # 绘制分割区域
            cv2.rectangle(image, (0, current_pos), (prompt_width, current_pos + region_height), colors[i % len(colors)], 2)
            current_pos += region_height

    return image


if __name__ == "__main__":
    bbox_inputs = [[5, 20, 100, 150], [160, 20, 190, 210], [230,5,290,290]]
    # bbox_inputs = [[40, 5, 210, 160], [100, 180, 180, 270]]
    prompt_width = 300
    prompt_height = 300

    HB_m_offset_list, HB_n_offset_list, HB_m_scale_list, HB_n_scale_list,SR_hw_split_ratio = generate_parameters(bbox_inputs, prompt_width, prompt_height)

    print("HB_m_offset_list:", HB_m_offset_list)
    print("HB_n_offset_list:", HB_n_offset_list)
    print("HB_m_scale_list:", HB_m_scale_list)
    print("HB_n_scale_list:", HB_n_scale_list)
    print("SR_hw_split_ratio:",SR_hw_split_ratio)