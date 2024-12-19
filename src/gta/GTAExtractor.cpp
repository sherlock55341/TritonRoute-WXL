#include "GTA.hpp"
#include <chrono>

namespace gta {
void GTA::extractGTADataFromDatabase() {
    extractTechDesignBasicInfo();
    extractIrInfo();
    extractBlkInfo();
    extractGCellInfo();
    extractCostInfo();
}

void GTA::extractTechDesignBasicInfo() {
    auto tp0 = std::chrono::high_resolution_clock::now();
    data.num_layers = tech->getLayers().size() - 2;
    for (auto &g : design->getTopBlock()->getGCellPatterns()) {
        // std::cout << g.isHorizontal() << " " << g.getStartCoord() << " "
        //           << g.getSpacing() << " " << g.getCount() << std::endl;
        if (g.isHorizontal() == 0) {
            data.num_gcells_x = g.getCount();
            data.gcell_start_x = g.getStartCoord();
            data.gcell_step_x = g.getSpacing();
        } else {
            data.num_gcells_y = g.getCount();
            data.gcell_start_y = g.getStartCoord();
            data.gcell_step_y = g.getSpacing();
        }
    }
    data.num_guides = 0;
    auto &nets = design->getTopBlock()->getNets();
    for (auto i = 0; i < nets.size(); i++) {
        auto net = nets[i].get();
        data.num_guides += net->getGuides().size();
    }
    data.num_nets = nets.size();
    data.layer_type = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_direction = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_width = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_pitch = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_track_start = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_track_step = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_track_num = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_wire_weight = (float *)malloc(sizeof(float) * data.num_layers);
    data.layer_eol_spacing = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_eol_width = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_eol_within = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_panel_start = (int *)calloc(sizeof(int), data.num_layers + 1);
    data.layer_panel_length = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_gcell_start = (int *)calloc(sizeof(int), data.num_layers + 1);
    data.layer_spacing_table_spacing_start =
        (int *)calloc(sizeof(int), data.num_layers + 1);
    data.layer_spacing_table_prl_start =
        (int *)calloc(sizeof(int), data.num_layers + 1);
    data.layer_spacing_table_width_start =
        (int *)calloc(sizeof(int), data.num_layers + 1);
    data.layer_via_lower_width = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_via_lower_length = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_via_upper_width = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_via_upper_length = (int *)malloc(sizeof(int) * data.num_layers);
    data.layer_enable_via_nbr_drc =
        (bool *)malloc(sizeof(bool) * data.num_layers);
    // for (auto i = 0; i < tech->getLayers().size(); i++){
    //     std::cout << i << " " << tech->getLayer(i)->getName() << std::endl;
    // }
    for (auto i = 0; i < data.num_layers; i++) {
        auto l = tech->getLayer(i + 2);
        data.layer_direction[i] = -1;
        data.layer_width[i] = -1;
        data.layer_pitch[i] = -1;
        data.layer_track_start[i] = -1;
        data.layer_track_step[i] = -1;
        data.layer_track_num[i] = -1;
        data.layer_wire_weight[i] = -1;
        data.layer_eol_spacing[i] = -1;
        data.layer_eol_width[i] = -1;
        data.layer_eol_within[i] = -1;
        data.layer_spacing_table_spacing_start[i + 1] = 0;
        data.layer_spacing_table_prl_start[i + 1] = 0;
        data.layer_spacing_table_width_start[i + 1] = 0;
        data.layer_via_lower_width[i] = -1;
        data.layer_via_lower_length[i] = -1;
        data.layer_via_upper_width[i] = -1;
        data.layer_via_upper_length[i] = -1;
        data.layer_enable_via_nbr_drc[i] = false;
        if (l->getType() == fr::frLayerTypeEnum::ROUTING) {
            data.layer_type[i] = 0;
            data.layer_direction[i] =
                (l->getDir() ==
                 fr::frPrefRoutingDirEnum::frcVertPrefRoutingDir);
            data.layer_width[i] = l->getWidth();
            data.layer_pitch[i] = l->getPitch();
            for (auto &tp : design->getTopBlock()->getTrackPatterns(i + 2)) {
                if (data.layer_direction[i] == (tp->isHorizontal() == true)) {
                    data.layer_track_start[i] = tp->getStartCoord();
                    data.layer_track_step[i] = tp->getTrackSpacing();
                    data.layer_track_num[i] = tp->getNumTracks();
                }
            }
            data.layer_wire_weight[i] = 1.0f;
            for (auto &con : l->getEolSpacing()) {
                if (con->getEolWidth() > l->getWidth()) {
                    data.layer_eol_spacing[i] = con->getMinSpacing();
                    data.layer_eol_width[i] = con->getEolWidth();
                    data.layer_eol_within[i] = con->getEolWithin();
                    break;
                }
            }
            data.layer_panel_start[i + 1] =
                (data.layer_direction[i] ? data.num_gcells_x
                                         : data.num_gcells_y);
            data.layer_panel_length[i] =
                (data.layer_direction[i] ? data.num_gcells_y
                                         : data.num_gcells_x);
            data.layer_gcell_start[i + 1] =
                data.num_gcells_x * data.num_gcells_y;
            // std::cout << data.layer_panel_length[i] << std::endl;
            auto con = l->getMinSpacing();
            assert(con);
            assert(con->typeId() ==
                   fr::frConstraintTypeEnum::frcSpacingTablePrlConstraint);
            auto prlCon = static_cast<fr::frSpacingTablePrlConstraint *>(con);
            auto &tbl = prlCon->getLookupTbl();
            auto row = tbl.getRows();
            auto col = tbl.getCols();
            data.layer_spacing_table_width_start[i + 1] = row.size();
            data.layer_spacing_table_prl_start[i + 1] = col.size();
            data.layer_spacing_table_spacing_start[i + 1] =
                row.size() * col.size();

        } else if (l->getType() == fr::frLayerTypeEnum::CUT) {
            data.layer_type[i] = 1;
            auto viaDef = l->getDefaultViaDef();
            assert(viaDef != nullptr);
            for (auto delta = -1; delta < 2; delta += 2) {
                fr::frVia via(viaDef);
                fr::frBox box(0, 0, 0, 0);
                if (delta > 0)
                    via.getLayer1BBox(box);
                else
                    via.getLayer2BBox(box);
                auto width = box.width();
                auto length = box.length();
                if (delta > 0) {
                    data.layer_via_upper_length[i] = length;
                    data.layer_via_upper_width[i] = width;
                } else {
                    data.layer_via_lower_length[i] = length;
                    data.layer_via_lower_width[i] = width;
                }
            }
        } else
            data.layer_type[i] = -1;
    }
    for (auto i = 0; i < data.num_layers; i++) {
        data.layer_panel_start[i + 1] += data.layer_panel_start[i];
        data.layer_gcell_start[i + 1] += data.layer_gcell_start[i];
        data.layer_spacing_table_width_start[i + 1] +=
            data.layer_spacing_table_width_start[i];
        data.layer_spacing_table_prl_start[i + 1] +=
            data.layer_spacing_table_prl_start[i];
        data.layer_spacing_table_spacing_start[i + 1] +=
            data.layer_spacing_table_spacing_start[i];
        if (data.layer_type[i] == 1) {
            assert(i + 1 >= 0 && i + 1 < tech->getLayers().size());
            assert(tech->getLayer(i + 1)->getType() ==
                   fr::frLayerTypeEnum::ROUTING);
            auto con = tech->getLayer(i + 1)->getMinSpacing();
            assert(con);
            assert(con->typeId() ==
                   fr::frConstraintTypeEnum::frcSpacingTablePrlConstraint);
            auto prlCon = static_cast<fr::frSpacingTablePrlConstraint *>(con);
            auto s = prlCon->find(std::max(data.layer_via_lower_width[i],
                                           data.layer_width[i - 1]),
                                  data.layer_via_lower_length[i]);
            if (s + data.layer_width[i - 1] / 2 +
                    data.layer_via_lower_width[i - 1] / 2 >
                data.layer_track_step[i - 1])
                data.layer_enable_via_nbr_drc[i - 1] = true;

            assert(i + 3 >= 0 && i + 3 < tech->getLayers().size());
            assert(tech->getLayer(i + 3)->getType() ==
                   fr::frLayerTypeEnum::ROUTING);
            con = tech->getLayer(i + 3)->getMinSpacing();
            assert(con);
            assert(con->typeId() ==
                   fr::frConstraintTypeEnum::frcSpacingTablePrlConstraint);
            prlCon = static_cast<fr::frSpacingTablePrlConstraint *>(con);
            s = prlCon->find(std::max(data.layer_via_upper_width[i],
                                      data.layer_width[i + 1]),
                             data.layer_via_upper_length[i]);
            if (s + data.layer_width[i + 1] / 2 +
                    data.layer_via_upper_width[i + 1] / 2 >
                data.layer_track_step[i + 1])
                data.layer_enable_via_nbr_drc[i + 1] = true;
        }
    }

    data.layer_spacing_table_width = (int *)malloc(
        sizeof(int) * data.layer_spacing_table_width_start[data.num_layers]);
    data.layer_spacing_table_prl = (int *)malloc(
        sizeof(int) * data.layer_spacing_table_prl_start[data.num_layers]);
    data.layer_spacing_table_spacing = (int *)malloc(
        sizeof(int) * data.layer_spacing_table_spacing_start[data.num_layers]);

    for (auto i = 0; i < data.num_layers; i++) {
        auto l = tech->getLayer(i + 2);
        auto con = l->getMinSpacing();
        if (con) {
            assert(con->typeId() ==
                   fr::frConstraintTypeEnum::frcSpacingTablePrlConstraint);
            auto prlCon = static_cast<fr::frSpacingTablePrlConstraint *>(con);
            auto &tbl = prlCon->getLookupTbl();
            auto row = tbl.getRows();
            auto col = tbl.getCols();
            auto vals = tbl.getVals();
            for (auto j = 0; j < row.size(); j++)
                data.layer_spacing_table_width
                    [data.layer_spacing_table_width_start[i] + j] = row[j];
            for (auto j = 0; j < col.size(); j++)
                data.layer_spacing_table_prl
                    [data.layer_spacing_table_prl_start[i] + j] = col[j];
            for (auto j = 0; j < row.size(); j++)
                for (auto k = 0; k < col.size(); k++)
                    data.layer_spacing_table_spacing
                        [data.layer_spacing_table_spacing_start[i] +
                         j * col.size() + k] = vals[j][k];
        }
    }
    for (auto i = 0; i < data.num_layers; i++)
        if (data.layer_type[i] == 0)
            std::cout << data.layer_enable_via_nbr_drc[i] << " ";
    std::cout << std::endl;
    auto tp1 = std::chrono::high_resolution_clock::now();
    std::cout << "extractTechDesignBasicInfo : "
              << std::chrono::duration_cast<std::chrono::microseconds>(tp1 -
                                                                       tp0)
                         .count() /
                     1e6
              << " s" << std::endl;
}

void GTA::extractIrInfo() {
    auto tp0 = std::chrono::high_resolution_clock::now();
    auto &nets = design->getTopBlock()->getNets();
    int *net_guide_offset = (int *)calloc(sizeof(int), nets.size() + 1);
    for (auto i = 0; i < nets.size(); i++) {
        auto net = nets[i].get();
        assert(i == net->getId());
        net_guide_offset[i + 1] = net->getGuides().size();
    }
    for (auto i = 0; i < nets.size(); i++)
        net_guide_offset[i + 1] += net_guide_offset[i];
    // std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    data.ir_layer = (short *)malloc(sizeof(short) * data.num_guides);
    data.ir_net = (int *)malloc(sizeof(int) * data.num_guides);
    data.ir_panel = (short *)malloc(sizeof(short) * data.num_guides);
    data.ir_gcell_begin = (short *)malloc(sizeof(short) * data.num_guides);
    data.ir_gcell_end = (short *)malloc(sizeof(short) * data.num_guides);
    data.ir_begin = (int *)malloc(sizeof(int) * data.num_guides);
    data.ir_end = (int *)malloc(sizeof(int) * data.num_guides);
    data.ir_track = (int *)malloc(sizeof(int) * data.num_guides);
    data.ir_track_low = (int *)malloc(sizeof(int) * data.num_guides);
    data.ir_track_high = (int *)malloc(sizeof(int) * data.num_guides);
    data.ir_wl_weight = (float *)malloc(sizeof(float) * data.num_guides);
    data.ir_has_ap = (bool *)malloc(sizeof(bool) * data.num_guides);
    data.ir_has_proj_ap = (bool *)malloc(sizeof(bool) * data.num_guides);
    data.ir_ap = (int *)malloc(sizeof(int) * data.num_guides);
    data.ir_nbr_start = (int *)calloc(sizeof(int), data.num_guides + 1);
    data.ir_reassign = (int *)calloc(sizeof(int), data.num_guides);
#pragma omp parallel for
    for (auto i = 0; i < nets.size(); i++) {
        auto net = nets[i].get();
        auto &guides = net->getGuides();
        for (auto j = 0; j < guides.size(); j++) {
            auto g = guides[j].get();
            auto g_idx = net_guide_offset[i] + j;
            fr::frPoint bp, ep;
            auto l = g->getBeginLayerNum() - 2;
            g->getPoints(bp, ep);
            data.ir_layer[g_idx] = l;
            assert(data.ir_layer[g_idx] >= 0 &&
                   data.ir_layer[g_idx] < data.num_layers &&
                   data.layer_type[data.ir_layer[g_idx]] == 0);
            data.ir_net[g_idx] = i;
            auto g_lx = (bp.x() - data.gcell_start_x) / data.gcell_step_x;
            auto g_ly = (bp.y() - data.gcell_start_y) / data.gcell_step_y;
            auto g_hx = (ep.x() - data.gcell_start_x) / data.gcell_step_x;
            auto g_hy = (ep.y() - data.gcell_start_y) / data.gcell_step_y;
            if (data.layer_direction[l]) {
                assert(g_lx == g_hx);
                data.ir_panel[g_idx] = g_lx;
                data.ir_gcell_begin[g_idx] = g_ly;
                data.ir_gcell_end[g_idx] = g_hy;
                auto lx = data.gcell_start_x + data.gcell_step_x * g_lx;
                auto hx = data.gcell_start_x + data.gcell_step_x * (g_lx + 1);
                data.ir_track_low[g_idx] =
                    std::ceil(1.0 * (lx - data.layer_track_start[l]) /
                              data.layer_track_step[l]);
                data.ir_track_low[g_idx] =
                    std::max(0, std::min(data.layer_track_num[l] - 1,
                                         data.ir_track_low[g_idx]));
                data.ir_track_high[g_idx] =
                    std::floor(1.0 * (hx - data.layer_track_start[l]) /
                               data.layer_track_step[l]);
                if (data.layer_track_start[l] + data.layer_track_step[l] *
                                                    data.ir_track_high[g_idx] ==
                        hx &&
                    data.ir_panel[g_idx] + 1 < data.num_gcells_x)
                    data.ir_track_high[g_idx]--;
            } else {
                assert(g_ly == g_hy);
                data.ir_panel[g_idx] = g_ly;
                data.ir_gcell_begin[g_idx] = g_lx;
                data.ir_gcell_end[g_idx] = g_hx;
                auto ly = data.gcell_start_y + data.gcell_step_y * g_ly;
                auto hy = data.gcell_start_y + data.gcell_step_y * (g_ly + 1);
                data.ir_track_low[g_idx] =
                    std::ceil(1.0 * (ly - data.layer_track_start[l]) /
                              data.layer_track_step[l]);
                data.ir_track_low[g_idx] =
                    std::max(0, std::min(data.layer_track_num[l] - 1,
                                         data.ir_track_low[g_idx]));
                data.ir_track_high[g_idx] =
                    std::floor(1.0 * (hy - data.layer_track_start[l]) /
                               data.layer_track_step[l]);
                if (data.layer_track_start[l] + data.layer_track_step[l] *
                                                    data.ir_track_high[g_idx] ==
                        hy &&
                    data.ir_panel[g_idx] + 1 < data.num_gcells_y)
                    data.ir_track_high[g_idx]--;
            }
            if (data.ir_track_high[g_idx] - data.ir_track_low[g_idx] > 15 ||
                data.ir_track_high[g_idx] - data.ir_track_low[g_idx] < 0) {
                std::cout << g_idx << " " << data.ir_track_low[g_idx] << " "
                          << data.ir_track_high[g_idx] << std::endl;
                std::cout << g_lx << " " << g_ly << " " << g_hx << " " << g_hy
                          << std::endl;
                std::cout << data.layer_track_start[l] << " "
                          << data.layer_track_step[l] << " "
                          << data.layer_track_num[l] << std::endl;
                exit(0);
            }
        }
    }
#pragma omp barrier
    // std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    std::vector<std::vector<int>> nbrGuideVec(data.num_guides);
#pragma omp parallel for
    for (auto i = 0; i < data.num_nets; i++) {
        auto net = nets[i].get();
        auto &guides = net->getGuides();
        for (auto j = 0; j < guides.size(); j++) {
            auto idx_1 = net_guide_offset[i] + j;
            for (auto k = j + 1; k < guides.size(); k++) {
                auto idx_2 = net_guide_offset[i] + k;
                if (std::abs(data.ir_layer[idx_1] - data.ir_layer[idx_2]) != 2)
                    continue;
                if (data.ir_panel[idx_1] >= data.ir_gcell_begin[idx_2] &&
                    data.ir_panel[idx_1] <= data.ir_gcell_end[idx_2] &&
                    data.ir_panel[idx_2] >= data.ir_gcell_begin[idx_1] &&
                    data.ir_panel[idx_2] <= data.ir_gcell_end[idx_1]) {
                    nbrGuideVec[idx_1].push_back(idx_2);
                    nbrGuideVec[idx_2].push_back(idx_1);
                }
            }
        }
    }
#pragma omp barrier
    // std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    for (auto i = 0; i < data.num_guides; i++)
        data.ir_nbr_start[i + 1] = nbrGuideVec[i].size();
    for (auto i = 0; i < data.num_guides; i++)
        data.ir_nbr_start[i + 1] += data.ir_nbr_start[i];
    data.ir_nbr_list =
        (int *)malloc(sizeof(int) * data.ir_nbr_start[data.num_guides]);
    for (auto i = 0; i < data.num_guides; i++)
        for (auto j = 0; j < nbrGuideVec[i].size(); j++)
            data.ir_nbr_list[data.ir_nbr_start[i] + j] = nbrGuideVec[i][j];
    nbrGuideVec.clear();
    nbrGuideVec.shrink_to_fit();
    // std::cout << __FILE__ << ":" << __LINE__ << std::endl;
#pragma omp parallel for
    for (auto i = 0; i < data.num_guides; i++) {
        data.ir_track[i] = -1;
        data.ir_wl_weight[i] = 0;
        data.ir_has_ap[i] = false;
        data.ir_has_proj_ap[i] = false;
        for (auto j = data.ir_nbr_start[i]; j < data.ir_nbr_start[i + 1]; j++) {
            auto nbr = data.ir_nbr_list[j];
            auto nbr_layer = data.ir_layer[nbr];
            if (data.ir_gcell_begin[nbr] < data.ir_panel[i])
                data.ir_wl_weight[i] -= data.layer_wire_weight[nbr_layer];
            if (data.ir_gcell_end[nbr] > data.ir_panel[i])
                data.ir_wl_weight[i] += data.layer_wire_weight[nbr_layer];
        }
    }
#pragma omp barrier
    // std::cout << __FILE__ << ":" << __LINE__ << std::endl;
#pragma omp parallel for
    for (auto i = 0; i < data.num_nets; i++) {
        auto net = nets[i].get();
        // std::cout << "finish " << i << std::endl;
        for (auto j = 0; j < net->getGuides().size(); j++) {
            auto g = net->getGuides()[j].get();
            auto g_idx = net_guide_offset[i] + j;
            if (findAp(g, g_idx) == false)
                findProjAp(g, g_idx);
        }
    }
#pragma omp barrier
    // std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    free(net_guide_offset);
    if (ENABLE_TA_VIA_DRC) {
        data.ir_gcell_begin_via_offset =
            (short *)malloc(sizeof(short) * data.num_guides);
        data.ir_gcell_end_via_offset =
            (short *)malloc(sizeof(short) * data.num_guides);
#pragma omp parallel for
        for (auto i = 0; i < data.num_guides; i++) {
            auto l = data.ir_layer[i];
            data.ir_gcell_begin_via_offset[i] = 0;
            data.ir_gcell_end_via_offset[i] = 0;
            if (data.layer_enable_via_nbr_drc[i] == false)
                continue;
            for (auto nbr_idx = data.ir_nbr_start[i];
                 nbr_idx < data.ir_nbr_start[i + 1]; nbr_idx++) {
                auto nbr = data.ir_nbr_list[nbr_idx];
                auto nbr_layer = data.ir_layer[nbr];
                auto via_layer = (l + nbr_layer) / 2;
                assert(data.layer_type[via_layer] == 1);
                auto width =
                    (nbr_layer < l ? data.layer_via_upper_width[via_layer]
                                   : data.layer_via_lower_width[via_layer]);
                auto length =
                    (nbr_layer < l ? data.layer_via_upper_length[via_layer]
                                   : data.layer_via_lower_length[via_layer]);
                auto s = findPRLSpacing(l, std::max(width, data.layer_width[l]),
                                        length);
                auto gap = s + width / 2 + data.layer_width[l] / 2;
                short delta = gap / data.layer_track_step[l];
                if (data.layer_track_step[l] * delta == gap)
                    delta--;
                if (data.ir_panel[nbr] == data.ir_gcell_begin[i])
                    data.ir_gcell_begin_via_offset[i] =
                        std::max(data.ir_gcell_begin_via_offset[i], delta);
                if (data.ir_panel[nbr] == data.ir_gcell_end[i])
                    data.ir_gcell_end_via_offset[i] =
                        std::max(data.ir_gcell_end_via_offset[i], delta);
            }
        }
#pragma omp barrier
    }
    auto tp1 = std::chrono::high_resolution_clock::now();
    std::cout << "extractIrInfo : "
              << std::chrono::duration_cast<std::chrono::microseconds>(tp1 -
                                                                       tp0)
                         .count() /
                     1e6
              << " s" << std::endl;
}

void GTA::extractBlkInfo() {
    auto tp0 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<fr::rq_rptr_value_t<fr::frBlockObject>>> results(
        data.num_layers);
    for (auto i = 0; i < data.num_layers; i++) {
        if (data.layer_type[i] != 0)
            continue;
        auto l = tech->getLayer(i + 2);
        auto lx = data.gcell_start_x;
        auto ly = data.gcell_start_y;
        auto hx =
            data.gcell_start_x + data.gcell_step_x * (data.num_gcells_x + 1);
        auto hy =
            data.gcell_start_y + data.gcell_step_y * (data.num_gcells_y + 1);
        fr::frBox box(lx, ly, hx, hy);
        design->getRegionQuery()->query(box, i + 2, results[i]);
    }
    data.num_blks = 0;
    for (auto i = 0; i < data.num_layers; i++)
        data.num_blks += results[i].size();
    data.b_left = (int *)malloc(sizeof(int) * data.num_blks);
    data.b_bottom = (int *)malloc(sizeof(int) * data.num_blks);
    data.b_right = (int *)malloc(sizeof(int) * data.num_blks);
    data.b_top = (int *)malloc(sizeof(int) * data.num_blks);
    data.b_net = (int *)malloc(sizeof(int) * data.num_blks);
    data.b_layer = (short *)malloc(sizeof(int) * data.num_blks);
    int *offset = (int *)calloc(sizeof(int), data.num_layers + 1);
    for (auto i = 0; i < data.num_layers; i++)
        offset[i + 1] = offset[i] + results[i].size();
    for (auto i = 0; i < data.num_layers; i++) {
        for (auto j = 0; j < results[i].size(); j++) {
            auto &box = results[i][j].first;
            auto obj = results[i][j].second;
            auto b = offset[i] + j;
            data.b_left[b] = box.min_corner().x();
            data.b_bottom[b] = box.min_corner().y();
            data.b_right[b] = box.max_corner().x();
            data.b_top[b] = box.max_corner().y();
            data.b_layer[b] = i;
            fr::frNet *netPtr = nullptr;
            if (obj->typeId() == fr::frcTerm)
                netPtr = static_cast<fr::frTerm *>(obj)->getNet();
            else if (obj->typeId() == fr::frcInstTerm)
                netPtr = static_cast<fr::frInstTerm *>(obj)->getNet();
            else if (obj->typeId() == fr::frcPathSeg)
                netPtr = static_cast<fr::frPathSeg *>(obj)->getNet();
            else if (obj->typeId() == fr::frcVia)
                netPtr = static_cast<fr::frVia *>(obj)->getNet();
            else if (obj->typeId() == fr::frcBlockage ||
                     obj->typeId() == fr::frcInstBlockage)
                netPtr = nullptr;
            else
                std::cout << __FILE__ << ":" << __LINE__ << " UNKNOWN TYPE"
                          << std::endl;
            data.b_net[b] = (netPtr ? netPtr->getId() : -1);
        }
    }
    free(offset);
    auto tp1 = std::chrono::high_resolution_clock::now();
    std::cout << "extractBlkInfo : "
              << std::chrono::duration_cast<std::chrono::microseconds>(tp1 -
                                                                       tp0)
                         .count() /
                     1e6
              << " s" << std::endl;
}

void GTA::extractGCellInfo() {
    auto tp0 = std::chrono::high_resolution_clock::now();
    data.gcell_end_point_ir_start =
        (int *)calloc(sizeof(int), data.layer_gcell_start[data.num_layers] + 1);
    for (auto i = 0; i < data.num_guides; i++) {
        auto idx =
            data.layer_gcell_start[data.ir_layer[i]] +
            data.ir_panel[i] * data.layer_panel_length[data.ir_layer[i]] +
            data.ir_gcell_begin[i];
        data.gcell_end_point_ir_start[idx + 1]++;
        if (data.ir_gcell_begin[i] != data.ir_gcell_end[i]) {
            idx += data.ir_gcell_end[i] - data.ir_gcell_begin[i];
            data.gcell_end_point_ir_start[idx + 1]++;
        }
    }
    for (auto i = 0; i < data.layer_gcell_start[data.num_layers]; i++)
        data.gcell_end_point_ir_start[i + 1] +=
            data.gcell_end_point_ir_start[i];
    int *offset = (int *)malloc(sizeof(int) *
                                (data.layer_gcell_start[data.num_layers] + 1));
    memcpy(offset, data.gcell_end_point_ir_start,
           sizeof(int) * (data.layer_gcell_start[data.num_layers] + 1));
    data.gcell_end_point_ir_list = (int *)malloc(
        sizeof(int) *
        data.gcell_end_point_ir_start[data.layer_gcell_start[data.num_layers]]);
    for (auto i = 0; i < data.num_guides; i++) {
        auto idx =
            data.layer_gcell_start[data.ir_layer[i]] +
            data.ir_panel[i] * data.layer_panel_length[data.ir_layer[i]] +
            data.ir_gcell_begin[i];
        data.gcell_end_point_ir_list[offset[idx]] = i;
        offset[idx]++;
        if (data.ir_gcell_begin[i] != data.ir_gcell_end[i]) {
            idx += data.ir_gcell_end[i] - data.ir_gcell_begin[i];
            data.gcell_end_point_ir_list[offset[idx]] = i;
            offset[idx]++;
        }
    }
    data.b_panel_begin = (short *)malloc(sizeof(short) * data.num_blks);
    data.b_panel_end = (short *)malloc(sizeof(short) * data.num_blks);
    data.b_gcell_begin = (short *)malloc(sizeof(short) * data.num_blks);
    data.b_gcell_end = (short *)malloc(sizeof(short) * data.num_blks);
    data.gcell_end_point_blk_start =
        (int *)calloc(sizeof(int), data.layer_gcell_start[data.num_layers] + 1);
    for (auto b = 0; b < data.num_blks; b++) {
        auto l = data.b_layer[b];
        auto isv = data.layer_direction[l];
        auto length = std::max(data.b_right[b] - data.b_left[b],
                               data.b_top[b] - data.b_bottom[b]);
        auto width = std::min(data.b_right[b] - data.b_left[b],
                              data.b_top[b] - data.b_bottom[b]);
        auto maxWidth = std::max(width, data.layer_width[l]);
        auto maxPRL = length;
        auto s = findPRLSpacing(l, maxWidth, maxPRL);
        if (ENABLE_TA_VIA_DRC) {
            if (l - 1 >= 0 && data.layer_type[l - 1] == 1) {
                maxWidth = std::max(width, data.layer_via_upper_width[l - 1]);
                maxPRL = std::min(length, data.layer_via_upper_length[l - 1]);
                s = std::max(s, findPRLSpacing(l, maxWidth, maxPRL));
            }
            if (l + 1 < data.num_layers && data.layer_type[l + 1] == 1) {
                maxWidth = std::max(width, data.layer_via_lower_width[l - 1]);
                maxPRL = std::min(length, data.layer_via_lower_length[l - 1]);
                s = std::max(s, findPRLSpacing(l, maxWidth, maxPRL));
            }
        }
        auto e_left = data.b_left[b] - s - data.layer_width[l] / 2 + 1;
        auto e_bottom = data.b_bottom[b] - s - data.layer_width[l] / 2 + 1;
        auto e_right = data.b_right[b] + s + data.layer_width[l] / 2 - 1;
        auto e_top = data.b_top[b] + s + data.layer_width[l] / 2 - 1;
        auto g_left = std::min(
            data.num_gcells_x - 1,
            std::max(0, (e_left - data.gcell_start_x) / data.gcell_step_x));
        auto g_bottom = std::min(
            data.num_gcells_y - 1,
            std::max(0, (e_bottom - data.gcell_start_y) / data.gcell_step_y));
        auto g_right = std::min(
            data.num_gcells_x - 1,
            std::max(0, (e_right - data.gcell_start_x) / data.gcell_step_x));
        auto g_top = std::min(
            data.num_gcells_y - 1,
            std::max(0, (e_top - data.gcell_start_y) / data.gcell_step_y));
        data.b_panel_begin[b] = (isv ? g_left : g_bottom);
        data.b_panel_end[b] = (isv ? g_right : g_top);
        data.b_gcell_begin[b] = (isv ? g_bottom : g_left);
        data.b_gcell_end[b] = (isv ? g_top : g_right);
        for (auto p = data.b_panel_begin[b]; p <= data.b_panel_end[b]; p++) {
            auto idx = data.layer_gcell_start[l] +
                       p * data.layer_panel_length[l] + data.b_gcell_begin[b];
            data.gcell_end_point_blk_start[idx + 1]++;
            if (data.b_gcell_begin[b] != data.b_gcell_end[b]) {
                idx += data.b_gcell_end[b] - data.b_gcell_begin[b];
                data.gcell_end_point_blk_start[idx + 1]++;
            }
        }
    }
    for (auto i = 0; i < data.layer_gcell_start[data.num_layers]; i++)
        data.gcell_end_point_blk_start[i + 1] +=
            data.gcell_end_point_blk_start[i];
    data.gcell_end_point_blk_list = (int *)malloc(
        sizeof(int) * data.gcell_end_point_blk_start
                          [data.layer_gcell_start[data.num_layers]]);
    memcpy(offset, data.gcell_end_point_blk_start,
           sizeof(int) * (data.layer_gcell_start[data.num_layers] + 1));
    for (auto b = 0; b < data.num_blks; b++) {
        auto l = data.b_layer[b];
        for (auto p = data.b_panel_begin[b]; p <= data.b_panel_end[b]; p++) {
            auto idx = data.layer_gcell_start[l] +
                       p * data.layer_panel_length[l] + data.b_gcell_begin[b];
            data.gcell_end_point_blk_list[offset[idx]] = b;
            offset[idx]++;
            if (data.b_gcell_begin[b] != data.b_gcell_end[b]) {
                idx += data.b_gcell_end[b] - data.b_gcell_begin[b];
                data.gcell_end_point_blk_list[offset[idx]] = b;
                offset[idx]++;
            }
        }
    }
    if (ENABLE_TA_VIA_DRC) {
        data.gcell_cross_ir_start = (int *)calloc(
            sizeof(int), data.layer_gcell_start[data.num_layers] + 1);
        for (auto i = 0; i < data.num_guides; i++) {
            auto l = data.ir_layer[i];
            for (auto g = data.ir_gcell_begin[i] + 1; g < data.ir_gcell_end[i];
                 g++) {
                auto idx = data.layer_gcell_start[l] +
                           data.ir_panel[i] * data.layer_panel_length[l] + g;
                data.gcell_cross_ir_start[idx + 1]++;
            }
        }
        for (auto i = 0; i < data.layer_gcell_start[data.num_layers]; i++)
            data.gcell_cross_ir_start[i + 1] += data.gcell_cross_ir_start[i];
        data.gcell_cross_ir_list = (int *)malloc(
            sizeof(int) * (data.gcell_cross_ir_start
                               [data.layer_gcell_start[data.num_layers]]));
        memcpy(offset, data.gcell_cross_ir_start,
               sizeof(int) * (data.layer_gcell_start[data.num_layers] + 1));
        for (auto i = 0; i < data.num_guides; i++) {
            auto l = data.ir_layer[i];
            for (auto g = data.ir_gcell_begin[i] + 1; g < data.ir_gcell_end[i];
                 g++) {
                auto idx = data.layer_gcell_start[l] +
                           data.ir_panel[i] * data.layer_panel_length[l] + g;
                data.gcell_cross_ir_list[offset[idx]] = i;
                offset[idx]++;
            }
        }
    }
    data.ir_super_set_start = (int *)calloc(sizeof(int), data.num_guides + 1);
    data.blk_super_set_start = (int *)calloc(sizeof(int), data.num_guides + 1);
    for (auto i = 0; i < data.num_guides; i++) {
        auto l = data.ir_layer[i];
        for (auto g = data.ir_gcell_begin[i] + 1; g <= data.ir_gcell_end[i] - 1;
             g++) {
            auto idx = data.layer_gcell_start[l] +
                       data.layer_panel_length[l] * data.ir_panel[i] + g;
            for (auto k = data.gcell_end_point_ir_start[idx];
                 k < data.gcell_end_point_ir_start[idx + 1]; k++) {
                auto j = data.gcell_end_point_ir_list[k];
                if (data.ir_gcell_end[j] == g &&
                    data.ir_gcell_begin[j] >= data.ir_gcell_begin[i] + 1)
                    data.ir_super_set_start[j + 1]++;
            }
        }
    }
    for (auto i = 0; i < data.num_blks; i++) {
        auto l = data.b_layer[i];
        for (auto p = data.b_panel_begin[i]; p <= data.b_panel_end[i]; p++) {
            for (auto g = data.b_gcell_begin[i] + 1;
                 g <= data.b_gcell_end[i] - 1; g++) {
                auto idx = data.layer_gcell_start[l] +
                           data.layer_panel_length[l] * p + g;
                for (auto k = data.gcell_end_point_ir_start[idx];
                     k < data.gcell_end_point_ir_start[idx + 1]; k++) {
                    auto j = data.gcell_end_point_ir_list[k];
                    if (data.ir_gcell_end[j] == g &&
                        data.ir_gcell_begin[j] >= data.b_gcell_begin[i] + 1)
                        data.blk_super_set_start[j + 1]++;
                }
            }
        }
    }
    for (auto i = 0; i < data.num_guides; i++) {
        data.ir_super_set_start[i + 1] += data.ir_super_set_start[i];
        data.blk_super_set_start[i + 1] += data.blk_super_set_start[i];
    }
    data.ir_super_set_list =
        (int *)malloc(sizeof(int) * data.ir_super_set_start[data.num_guides]);
    data.blk_super_set_list =
        (int *)malloc(sizeof(int) * data.blk_super_set_start[data.num_guides]);
    offset = (int *)realloc(offset, sizeof(int) * (data.num_guides + 1));
    memcpy(offset, data.ir_super_set_start,
           sizeof(int) * (data.num_guides + 1));
    for (auto i = 0; i < data.num_guides; i++) {
        auto l = data.ir_layer[i];
        for (auto g = data.ir_gcell_begin[i] + 1; g <= data.ir_gcell_end[i] - 1;
             g++) {
            auto idx = data.layer_gcell_start[l] +
                       data.layer_panel_length[l] * data.ir_panel[i] + g;
            for (auto k = data.gcell_end_point_ir_start[idx];
                 k < data.gcell_end_point_ir_start[idx + 1]; k++) {
                auto j = data.gcell_end_point_ir_list[k];
                if (data.ir_gcell_end[j] == g &&
                    data.ir_gcell_begin[j] >= data.ir_gcell_begin[i] + 1) {
                    data.ir_super_set_list[offset[j]] = i;
                    offset[j]++;
                }
            }
        }
    }
    memcpy(offset, data.blk_super_set_start,
           sizeof(int) * (data.num_guides + 1));
    for (auto i = 0; i < data.num_blks; i++) {
        auto l = data.b_layer[i];
        for (auto p = data.b_panel_begin[i]; p <= data.b_panel_end[i]; p++) {
            for (auto g = data.b_gcell_begin[i] + 1;
                 g <= data.b_gcell_end[i] - 1; g++) {
                auto idx = data.layer_gcell_start[l] +
                           data.layer_panel_length[l] * p + g;
                for (auto k = data.gcell_end_point_ir_start[idx];
                     k < data.gcell_end_point_ir_start[idx + 1]; k++) {
                    auto j = data.gcell_end_point_ir_list[k];
                    if (data.ir_gcell_end[j] == g &&
                        data.ir_gcell_begin[j] >= data.b_gcell_begin[i] + 1) {
                        data.blk_super_set_list[offset[j]] = i;
                        offset[j]++;
                    }
                }
            }
        }
    }
    free(offset);
    auto tp1 = std::chrono::high_resolution_clock::now();
    std::cout << "extractGCellInfo : "
              << std::chrono::duration_cast<std::chrono::microseconds>(tp1 -
                                                                       tp0)
                         .count() /
                     1e6
              << " s" << std::endl;
}

void GTA::extractCostInfo() {
    data.ir_vio_cost_start = (int *)calloc(sizeof(int), data.num_guides + 1);
    for (auto i = 0; i < data.num_guides; i++)
        data.ir_vio_cost_start[i + 1] =
            data.ir_track_high[i] - data.ir_track_low[i] + 1;
    for (auto i = 0; i < data.num_guides; i++)
        data.ir_vio_cost_start[i + 1] += data.ir_vio_cost_start[i];
    data.ir_vio_cost_list =
        (int *)calloc(sizeof(int), data.ir_vio_cost_start[data.num_guides]);
    data.ir_align_list = (int8_t *)calloc(
        sizeof(int8_t), data.ir_vio_cost_start[data.num_guides]);
    data.ir_key_cost = (int *)calloc(sizeof(int), data.num_guides);
}

bool GTA::findAp(fr::frGuide *g, int g_idx) {
    fr::frPoint bp, ep;
    g->getPoints(bp, ep);
    if (bp != ep)
        return false;
    auto net = g->getNet();
    fr::frBox box;
    box.set(bp, bp);
    std::vector<fr::frBlockObject *> result;
    design->getRegionQuery()->queryGRPin(box, result);
    // find access point
    fr::frTransform instXform, shiftXform;
    fr::frTerm *trueTerm = nullptr;
    for (auto &term : result) {
        fr::frInst *inst = nullptr;
        if (term->typeId() == fr::frcInstTerm) {
            if (static_cast<fr::frInstTerm *>(term)->getNet() != net) {
                continue;
            }
            inst = static_cast<fr::frInstTerm *>(term)->getInst();
            inst->getTransform(shiftXform);
            shiftXform.set(fr::frOrient(fr::frcR0));
            inst->getUpdatedXform(instXform);
            trueTerm = static_cast<fr::frInstTerm *>(term)->getTerm();
        } else if (term->typeId() == fr::frcTerm) {
            if (static_cast<fr::frTerm *>(term)->getNet() != net) {
                continue;
            }
            trueTerm = static_cast<fr::frTerm *>(term);
        }
        if (trueTerm) {
            int pinIdx = 0;
            int paIdx = inst ? inst->getPinAccessIdx() : -1;
            for (auto &pin : trueTerm->getPins()) {
                fr::frAccessPoint *ap = nullptr;
                if (inst)
                    ap = (static_cast<fr::frInstTerm *>(term)
                              ->getAccessPoints())[pinIdx];
                if (!pin->hasPinAccess())
                    continue;
                if (paIdx == -1)
                    continue;
                if (ap == nullptr)
                    continue;
                fr::frPoint apBp;
                ap->getPoint(apBp);
                apBp.transform(shiftXform);
                if (ap->getLayerNum() - 2 == data.ir_layer[g_idx]) {
                    data.ir_has_ap[g_idx] = true;
                    data.ir_has_proj_ap[g_idx] = true;
                    auto is_v = data.layer_direction[data.ir_layer[g_idx]];
                    data.ir_ap[g_idx] = is_v ? apBp.x() : apBp.y();
                    data.ir_begin[g_idx] = is_v ? apBp.y() : apBp.x();
                    data.ir_end[g_idx] = is_v ? apBp.y() : apBp.x();
                    data.ir_wl_weight[g_idx] = 0;
                    return true;
                }
                pinIdx++;
            }
        }
    }
    return false;
}

void GTA::findProjAp(fr::frGuide *g, int g_idx) {
    fr::frPoint bp, ep;
    g->getPoints(bp, ep);
    auto net = g->getNet();
    fr::frBox box;
    box.set(bp, bp);
    std::vector<fr::frBlockObject *> result;
    design->getRegionQuery()->queryGRPin(box, result);
    if (ep != bp) {
        box.set(ep, ep);
        design->getRegionQuery()->queryGRPin(box, result);
    }
    // find access point
    fr::frTransform instXform, shiftXform;
    fr::frTerm *trueTerm = nullptr;
    for (auto &term : result) {
        fr::frInst *inst = nullptr;
        if (term->typeId() == fr::frcInstTerm) {
            if (static_cast<fr::frInstTerm *>(term)->getNet() != net) {
                continue;
            }
            inst = static_cast<fr::frInstTerm *>(term)->getInst();
            inst->getTransform(shiftXform);
            shiftXform.set(fr::frOrient(fr::frcR0));
            inst->getUpdatedXform(instXform);
            trueTerm = static_cast<fr::frInstTerm *>(term)->getTerm();
        } else if (term->typeId() == fr::frcTerm) {
            if (static_cast<fr::frTerm *>(term)->getNet() != net) {
                continue;
            }
            trueTerm = static_cast<fr::frTerm *>(term);
        }
        if (trueTerm) {
            int pinIdx = 0;
            int paIdx = inst ? inst->getPinAccessIdx() : -1;
            for (auto &pin : trueTerm->getPins()) {
                fr::frAccessPoint *ap = nullptr;
                if (inst)
                    ap = (static_cast<fr::frInstTerm *>(term)
                              ->getAccessPoints())[pinIdx];
                if (!pin->hasPinAccess())
                    continue;
                if (paIdx == -1)
                    continue;
                if (ap == nullptr)
                    continue;
                fr::frPoint apBp;
                ap->getPoint(apBp);
                apBp.transform(shiftXform);
                data.ir_has_proj_ap[g_idx] = true;
                auto is_v = data.layer_direction[data.ir_layer[g_idx]];
                data.ir_ap[g_idx] = is_v ? apBp.x() : apBp.y();
                return;
                pinIdx++;
            }
        }
        data.ir_ap[g_idx] = 0;
    }
    return;
}
} // namespace gta