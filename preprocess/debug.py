min_span_len = 5
import regex as re


def decide_segment(cur_segment, in_idx, in_lines):
    tmp_merged = cur_segment + in_lines[in_idx]

    tmp_merged_len = len(tmp_merged.strip().split())
    tmp_cur_segment_len = len(cur_segment.strip().split())
    tmp_next_seg_len = len(in_lines[in_idx].strip().split())

    if in_idx == len(in_lines) - 1:
        return True

    if in_lines[in_idx].startswith("`s ") or in_lines[in_idx].startswith("\'s ") or len(in_lines[in_idx]) < 3:
        return False

    if tmp_next_seg_len < min_span_len and in_lines[in_idx].endswith(". ") and (not cur_segment.endswith(". ")):
        return False

    if tmp_merged.endswith(": ") and len(tmp_merged) > 4 and tmp_merged.strip()[-3].isdigit() and in_idx + 1 < len(in_lines) and in_lines[in_idx + 1].strip()[0].isdigit():
        return False

    if in_lines[in_idx].endswith("said. ") or in_lines[in_idx].endswith("says. "):
        return True

    if len(re.findall("\`\s\`", tmp_merged)) > 0 and len(re.findall("\'\s\'", tmp_merged)) < 1:
        if (" says " not in cur_segment) and (" said " not in cur_segment):
            return False

    if tmp_merged.endswith(". "):
        return True

    if tmp_merged_len < min_span_len + 1:
        return False


    return True


def EDU_level_merge_segments(input_lines):
    input_lines = [i.replace("\n", " ").replace("` '", "' '") for i in input_lines]

    new_lines = []
    seg_idx = 0
    current_seg = ""

    while seg_idx < len(input_lines):
        current_seg = re.sub("\s+", " ", current_seg)

        if len(input_lines[seg_idx].strip().split()) < min_span_len and input_lines[seg_idx].endswith(". ") and len(new_lines) > 1 and (not new_lines[-1].endswith(". ")):
            new_lines[-1] = new_lines[-1] + current_seg + input_lines[seg_idx]

        else:
            if decide_segment(current_seg, seg_idx, input_lines) is False:
                current_seg = current_seg + input_lines[seg_idx]
            else:
                new_lines.append(current_seg + input_lines[seg_idx])
                current_seg = ""

        seg_idx += 1
    return new_lines


if __name__ == '__main__':
    orig = [' "(I) came in, ', ' I put my finger in her hand, ', ' and I told her ', ' her ', ' Dad was here, ', ' and I love her," ', ' he told reporters Wednesday. ']

    merged = EDU_level_merge_segments(orig)
    print(merged[-1])
